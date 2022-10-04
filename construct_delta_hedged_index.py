import sys
import pandas as pd
import numpy as np
import psycopg2


def getWRDSConnection():
    with open("account_data/wrds_user.txt") as f:
        wrds_username = f.readline()

    with open("account_data/wrds_pass.txt") as f:
        wrds_password = f.readline()

    conn = psycopg2.connect(
        host="wrds-pgdata.wharton.upenn.edu",
        port = 9737,
        database="wrds",
        user=wrds_username,
        password=wrds_password)

    return conn

def loadCRSPPermnoList(conn, permno_list):
    query_crsp = """
        select permno, date, prc, ret 
        from crsp.dsf 
        where permno in (__permno_list__) 
        and date >= '1996-01-01'
    """
    query_crsp = query_crsp.replace("__permno_list__", ", ".join([str(int(x)) for x in permno_list]))
    query_crsp = query_crsp.replace("\n", " ").strip()
    crsp = pd.read_sql_query(query_crsp, conn)
    crsp["date"] = pd.to_datetime(crsp["date"])

    # Calculating cumulative return
    crsp.sort_values(["permno", "date"], inplace=True)
    crsp = crsp[crsp["ret"].notnull()]
    crsp["logret"] = np.log(1 + crsp["ret"])
    crsp["cumlogret"] = crsp.groupby(["permno"])["logret"].cumsum()
    crsp["cumret"] = np.exp(crsp["cumlogret"])
    crsp.drop(columns=["logret", "cumlogret"], inplace=True)

    return crsp

def linkOptionMetricsCRSP(crsp, om_crsp_link):
    crsp = pd.merge(crsp, om_crsp_link, on="permno")
    crsp = crsp[(crsp["date"] >= crsp["sdate"]) & (crsp["date"] <= crsp["edate"])]
    crsp.drop(columns=["sdate", "edate"], inplace=True)
    return crsp

def getAllLastTradingDates(conn):
    # Getting full list of trading days from since Jan 1 1996
    # and finding the last trading date for each month
    query_dates = "select distinct date from crsp.dsf where date >= '1996-01-01'"
    full_date_list = pd.read_sql_query(query_dates, conn)

    full_date_list["date"] = pd.to_datetime(full_date_list["date"])
    full_date_list["form_month"] = full_date_list["date"] + pd.offsets.MonthEnd(0)
    full_date_list["next_month"] = full_date_list["form_month"] + pd.offsets.MonthEnd(1)
    last_trading_date = full_date_list[["date", "form_month"]].drop_duplicates()
    last_trading_date = last_trading_date.sort_values(["form_month", "date"]).groupby("form_month").last().reset_index()
    last_trading_date.rename(columns={"date": "form_last_date"}, inplace=True)
    last_trading_date.sort_values(["form_month"], inplace=True)
    last_trading_date["next_last_date"] = last_trading_date["form_last_date"].shift(-1)

    return last_trading_date

def formDeltaHedgedPositions(om, crsp, last_trading_date, target_delta=50, min_mat=60):
    # 1. For each month getting the last trading date using CRSP
    # to filter for options that on two last trading dates
    options_traded_on_last_date = pd.merge(
        last_trading_date, om[["optionid", "date", "mid"]], 
        left_on="form_last_date", right_on="date"
    ).drop(columns="date").rename(columns={"mid": "form_mid"})
    options_traded_on_last_date = pd.merge(
        options_traded_on_last_date, om[["optionid", "date", "mid"]],
        left_on=["optionid", "next_last_date"], right_on=["optionid", "date"]
    ).drop(columns="date").rename(columns={"mid": "next_mid"})
    options_traded_on_last_date = options_traded_on_last_date[
        options_traded_on_last_date["form_mid"].notnull() & 
        options_traded_on_last_date["next_mid"].notnull()
    ]

    # 2. Among these options, picking the ones with appropriate maturity
    # that is at least 30 days. Maybe if go further will get more options...
    options_traded_on_last_date = pd.merge(
        options_traded_on_last_date, om[["optionid", "date", "exdate"]], 
        left_on=["optionid", "form_last_date"], right_on=["optionid", "date"]
    ).drop(columns="date")

    date_exdate = options_traded_on_last_date[["form_last_date", "exdate"]].drop_duplicates()
    date_exdate = date_exdate.sort_values(["form_last_date", "exdate"])
    date_exdate["mat"] = (date_exdate["exdate"] - date_exdate["form_last_date"]).dt.days
    date_exdate = date_exdate[date_exdate["mat"] >= min_mat]
    date_exdate = date_exdate.groupby(["form_last_date"]).first().reset_index()
    date_exdate = date_exdate.drop(columns="mat").rename(columns={"exdate": "exdate_first"})

    target_options_step1 = pd.merge(
        date_exdate, options_traded_on_last_date, how="left", 
        left_on=["form_last_date", "exdate_first"], right_on=["form_last_date", "exdate"]
    ).drop(columns="exdate_first")

    # 4. Merging with options data on form date to get cp-flag and delta
    # to pick the appropriate option
    target_options_step1 = pd.merge(
        target_options_step1, om[["optionid", "date", "cp_flag", "delta"]],
        left_on=["form_last_date", "optionid"], right_on=["date", "optionid"]
    ).drop(columns="date")
    target_options_step1["ddelta"] = np.abs(np.abs(target_options_step1["delta"])*100.0 - target_delta)
    target_options_step1.sort_values(["form_last_date", "cp_flag", "ddelta"], inplace=True)
    target_options_step2 = target_options_step1.groupby(["form_last_date", "cp_flag"]).first().reset_index()
    target_options_step2.rename(columns={"delta": "form_delta"}, inplace=True)
    target_options_step2.drop(columns="ddelta", inplace=True)

    # 3. Merging with full options dataframe to get deltas and mid prices
    # over the holding periods of the option
    target_options_pl = pd.merge(
        target_options_step2, om[["optionid", "date", "date_month", "delta", "rf", "mid"]],
        left_on=["optionid"], right_on=["optionid"]
    )
    target_options_pl = pd.merge(
        target_options_pl, crsp[["date", "permno", "cumret", "prc"]],
        on="date", how = "left"
    )

    # Lagging variables to calculate gains from delta hedgind and funding costs
    target_options_pl.sort_values(["optionid", "date"], inplace=True)
    target_options_pl["cumret_lag"] = target_options_pl.groupby(["optionid"])["cumret"].shift(1)
    target_options_pl["prc_lag"] = target_options_pl.groupby(["optionid"])["prc"].shift(1)
    target_options_pl["delta_lag"] = target_options_pl.groupby(["optionid"])["delta"].shift(1)
    target_options_pl["rf_lag"] = target_options_pl.groupby(["optionid"])["rf"].shift(1)
    target_options_pl["date_lag"] = target_options_pl.groupby(["optionid"])["date"].shift(1)
    target_options_pl = target_options_pl[target_options_pl["form_month"] + pd.offsets.MonthEnd(1) == target_options_pl["date_month"]]

    # Gains from delta hedging and funding costs
    target_options_pl["dhedge_gain"] = -target_options_pl["delta_lag"]*((target_options_pl["cumret"]/target_options_pl["cumret_lag"])*target_options_pl["prc_lag"] - target_options_pl["prc_lag"])
    target_options_pl["funding_cost"] = -(target_options_pl["date"] - target_options_pl["date_lag"]).dt.days/365.0*target_options_pl["rf_lag"]*(target_options_pl["mid"] - target_options_pl["delta_lag"]*target_options_pl["prc_lag"])

    # 4. Aggregting delta hedged gains and funding costs and adding spot
    # price on the formation date
    sum_pl = target_options_pl.groupby(
        ["form_month", "form_last_date", "cp_flag", "optionid"]
    )[
        ["form_mid", "next_mid", "dhedge_gain", "funding_cost", "form_delta"]
    ].mean().reset_index()

    sum_pl_nobs = target_options_pl.groupby(
        ["form_month", "form_last_date", "cp_flag", "optionid"]
    )[
        ["date"]
    ].count().reset_index().rename(columns={"date": "nobs"})
    sum_pl = pd.merge(sum_pl, sum_pl_nobs, on=["form_month", "form_last_date", "cp_flag", "optionid"])

    # Getting the spot price at formation to have a reference for gains
    sum_pl = pd.merge(
        sum_pl, crsp[["date", "prc"]].rename(columns={"prc": "form_spot"}),
        left_on="form_last_date", right_on="date", how="left").drop(columns="date")

    sum_pl["gain"] = (sum_pl["next_mid"] - sum_pl["form_mid"]) + sum_pl["dhedge_gain"] + sum_pl["funding_cost"]
    sum_pl["gain_to_spot"] = sum_pl["gain"]/sum_pl["form_spot"]
    sum_pl["next_month"] = sum_pl["form_month"] + pd.offsets.MonthEnd(1)

    return sum_pl


def loadOptionData(appendix):
    # filename = f"option_data_update/opt_data_liquidity_{appendix}.csv"
    filename = f"option_data_update/opt_data_{appendix}.csv"
    om = pd.read_csv(filename)
    om["date"] = pd.to_datetime(om["date"])
    om["exdate"] = pd.to_datetime(om["exdate"])
    om["mid"] = 0.5*(om["best_offer"] + om["best_bid"])
    om["date_month"] = om["date"] + pd.offsets.MonthEnd(0)

    # Adding interest rate
    zcb = pd.read_csv("option_data/zcb_data.csv")
    zcb["date"] = pd.to_datetime(zcb["date"])
    zcb["rate"] = zcb["rate"]/100.0

    # Interpolating to 45 days for each date
    def interpolateZCB(subdf):
        return np.interp(45.0, subdf["days"], subdf["rate"])

    zcb30 = zcb.groupby("date").apply(interpolateZCB).rename("rf").reset_index()

    om = pd.merge(om, zcb30, on="date")

    # For some reason need to force variables to be of numeric type
    om["delta"] = pd.to_numeric(om["delta"])
    om["mid"] = pd.to_numeric(om["mid"])
    om["optionid"] = pd.to_numeric(om["optionid"])

    return om


from construct_delta_hedged_options import getWRDSConnection, getAllLastTradingDates, loadOptionData


# some prep work
conn = getWRDSConnection()
last_trading_date = getAllLastTradingDates(conn)

# Option data
om = loadOptionData("spx")

# Loading dividend yield data from OM
query = "select * from optionm.idxdvd where secid = 108105"
div_yield = pd.read_sql_query(query, conn)
div_yield.sort_values("date", inplace=True)
div_yield["rate"] = div_yield["rate"]/100.0
div_yield["date"] = pd.to_datetime(div_yield["date"])

# Calculating cumulative cum-dividend price process for S&P
cumret = om[["date", "spot"]].groupby("date").first().reset_index()
cumret = pd.merge(cumret, div_yield, on="date", how="left")
cumret["div"] = (cumret["spot"]*cumret["rate"]/365.0).shift(1)
cumret["ret"] = (cumret["spot"] + cumret["div"])/cumret["spot"].shift(1) - 1.0
cumret["logret"] = np.log(1 + cumret["ret"])
cumret["cumlogret"] = cumret["logret"].cumsum()
cumret["cumret"] = np.exp(cumret["cumlogret"])

# Converting to an appropriate format
cumret.rename(columns={"spot": "prc"}, inplace=True)
cumret = cumret[["date", "prc", "cumret"]]
cumret["cumret"] = cumret["cumret"].fillna(1.0)
cumret["permno"] = "S&P"

# Using the previous function in a hope that it works
pl_50 = formDeltaHedgedPositions(om, cumret, last_trading_date, target_delta=50, min_mat=60)
pl_50.to_csv("../data/delta_hedged_pl/delta_hedged_pl_spx.csv", index=False)

# # some testing
# df = pl_50.pivot_table(
#     index="form_month", columns="cp_flag", values="gain_to_spot"
# )
# df["gain_to_spot"] = df["C"] + df["P"]

# np.log(1 + df["gain_to_spot"]).cumsum().plot()




    