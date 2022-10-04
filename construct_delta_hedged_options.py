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
        select permno, date, prc, ret, bid, ask, (ask - bid) as ba_spread
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

    # Sometimes bid-ask spread is reported < 0. Make it zero then
    crsp["ba_spread"] = np.maximum(crsp["ba_spread"], 0.0)

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
        last_trading_date, om[["optionid", "date", "mid", "opt_ba_spread"]], 
        left_on="form_last_date", right_on="date"
    ).drop(columns="date").rename(columns={"mid": "form_mid", "opt_ba_spread": "form_opt_ba_spread"})
    options_traded_on_last_date = pd.merge(
        options_traded_on_last_date, om[["optionid", "date", "mid", "opt_ba_spread"]],
        left_on=["optionid", "next_last_date"], right_on=["optionid", "date"]
    ).drop(columns="date").rename(columns={"mid": "next_mid", "opt_ba_spread": "next_opt_ba_spread"})
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
        target_options_pl, crsp[["date", "permno", "cumret", "prc", "bid", "ask", "ba_spread"]],
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

    ####################################################################
    # Alternative specification for delta hedge gains
    target_options_pl["dhedge_cumgain"] = (1 + target_options_pl["delta_lag"]*(1 - target_options_pl["cumret"]/target_options_pl["cumret_lag"]))
    target_options_pl["dhedge_cumgain"] = target_options_pl.groupby(
        ["form_month", "form_last_date",  "next_last_date", "cp_flag", "optionid"]
    )["dhedge_cumgain"].cumprod() - 1.0
    ####################################################################

    # Accounting for delta-hedge trading costs
    target_options_pl["ddelta_abs"] = np.abs(target_options_pl["delta"] - target_options_pl["delta_lag"])
    target_options_pl["dhedge_tc"] = target_options_pl["ddelta_abs"]*0.5*target_options_pl["ba_spread"]

    # 4. Aggregting delta hedged gains and funding costs and adding spot
    # price on the formation date
    sum_opt_prices = target_options_pl.groupby(
        ["form_month", "form_last_date",  "next_last_date", "cp_flag", "optionid"]
    )[
        ["form_mid", "next_mid", "form_opt_ba_spread", "next_opt_ba_spread", "form_delta"]
    ].mean().reset_index()

    sum_pl_nobs = target_options_pl.groupby(
        ["form_month", "form_last_date",  "next_last_date", "cp_flag", "optionid"]
    )[
        ["date"]
    ].count().reset_index().rename(columns={"date": "nobs"})

    sum_pl = target_options_pl.groupby(
        ["form_month", "form_last_date",  "next_last_date", "cp_flag", "optionid"]
    )[
        ["dhedge_gain", "funding_cost", "dhedge_tc"]
    ].sum().reset_index()

    final_delta = target_options_pl.sort_values(
        ["form_month", "form_last_date",  "next_last_date", "cp_flag", "optionid", "date"]
    ).groupby(
        ["form_month", "form_last_date",  "next_last_date", "cp_flag", "optionid"]
    )[
        ["delta", "ddelta_abs", "ba_spread", "dhedge_cumgain"]
    ].last().reset_index().rename(columns={"delta": "last_delta", "ddelta_abs": "last_ddelta_abs", "ba_spread": "last_ba_spread"})

    sum_pl = pd.merge(sum_pl, sum_pl_nobs, on=["form_month", "form_last_date",  "next_last_date", "cp_flag", "optionid"])
    sum_pl = pd.merge(sum_pl, sum_opt_prices, on=["form_month", "form_last_date",  "next_last_date", "cp_flag", "optionid"])
    sum_pl = pd.merge(sum_pl, final_delta, on=["form_month", "form_last_date",  "next_last_date", "cp_flag", "optionid"])

    # Getting the spot price at formation to have a reference for gains
    sum_pl = pd.merge(
        sum_pl, crsp[["date", "prc", "ba_spread"]].rename(columns={"prc": "form_spot", "ba_spread": "form_ba_spread"}),
        left_on="form_last_date", right_on="date", how="left").drop(columns="date")

    # Getting spot price at the last date
    sum_pl = pd.merge(
        sum_pl, crsp[["date", "prc", "ba_spread"]].rename(columns={"prc": "next_spot", "ba_spread": "next_ba_spread"}),
        left_on="next_last_date", right_on="date", how="left").drop(columns="date")

    sum_pl["gain"] = (sum_pl["next_mid"] - sum_pl["form_mid"]) + sum_pl["dhedge_gain"] + sum_pl["funding_cost"]
    sum_pl["gain_to_spot"] = sum_pl["gain"]/sum_pl["form_spot"]
    sum_pl["next_month"] = sum_pl["form_month"] + pd.offsets.MonthEnd(1)

    # Removing the last transaction cost from the calculation
    sum_pl["dhedge_tc"] = sum_pl["dhedge_tc"] - sum_pl["last_ddelta_abs"]*0.5*sum_pl["last_ba_spread"]

    # Calculating total transaction costs from an option position
    sum_pl["stock_tc"] = 0.5*sum_pl["form_ba_spread"]*np.abs(sum_pl["form_delta"])
    sum_pl["stock_tc"] = sum_pl["stock_tc"] + 0.5*sum_pl["next_ba_spread"]*np.abs(sum_pl["last_delta"])

    sum_pl["opt_tc"] = 0.5*sum_pl["form_opt_ba_spread"]
    sum_pl["opt_tc"] = sum_pl["opt_tc"] + 0.5*sum_pl["next_opt_ba_spread"]

    return sum_pl #, target_options_pl


def loadOptionData(appendix):
    # filename = f"option_data_update/opt_data_liquidity_{appendix}.csv"
    filename = f"option_data_update/opt_data_{appendix}.csv"
    om = pd.read_csv(filename)
    om["date"] = pd.to_datetime(om["date"])
    om["exdate"] = pd.to_datetime(om["exdate"])
    om["mid"] = 0.5*(om["best_offer"] + om["best_bid"])
    om["opt_ba_spread"] = om["best_offer"] - om["best_bid"]
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


def main(argv = None):
    appendix = argv[1]

    # Loading file with options
    om = loadOptionData(appendix)

    # This will be an imput into the script
    secid_list = list(om["secid"].drop_duplicates())

    # Getting permno list from the linking file
    om_crsp_link = pd.read_csv("om_crsp_wrds_linking_table.csv")
    om_crsp_link = om_crsp_link[om_crsp_link["score"] < 6].drop(columns="score")
    om_crsp_link.rename(columns={"PERMNO": "permno"}, inplace=True)
    for col in ["sdate", "edate"]:
        om_crsp_link[col] = om_crsp_link[col].astype(int).astype(str)
        om_crsp_link[col] = pd.to_datetime(om_crsp_link[col])

    om_crsp_link_current = om_crsp_link[om_crsp_link["secid"].isin(secid_list)]
    permno_list = list(om_crsp_link_current["permno"])
    permno_list = list(set(permno_list))

    # Loading om, crsp and linking them
    conn = getWRDSConnection()
    crsp = loadCRSPPermnoList(conn, permno_list)
    crsp = linkOptionMetricsCRSP(crsp, om_crsp_link_current)
    last_trading_date = getAllLastTradingDates(conn)

    print("Calculating delta hedged strategy gains")
    sum_pl_50_list = []
    for isecid, secid in enumerate(secid_list):
        print(f" -- secid {isecid+1} out of {len(secid_list)}")
        sum_pl_50 = formDeltaHedgedPositions(
            om[om["secid"] == secid], 
            crsp[crsp["secid"] == secid], 
            last_trading_date, target_delta=50, min_mat=60
        )
        sum_pl_50["secid"] = secid
        sum_pl_50_list.append(sum_pl_50)

    sum_pl_50 = pd.concat(sum_pl_50_list, ignore_index=True)

    # Saving
    sum_pl_50.to_csv(f"delta_hedged_pl/delta_hedged_pl_50_{appendix}_update2.csv", index=False)

    conn.close()

if __name__ == "__main__":
    main(sys.argv)