from __future__ import print_function
from __future__ import division
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
import psycopg2
import time

def getOptionMetricsQuery():
    return """
        with security_table as (
            select
                secid, date, close as spot
            from optionm.SECPRD
            where secid = _secid_
            and date >= _start_date_
            and date <= _end_date_
        )
        select
            o.optionid, o.secid, o.date, o.exdate, o.cp_flag, o.strike_price/1000 as strike, 
            o.impl_volatility, o.best_offer, o.best_bid, o.delta,
            s.spot, o.volume, o.open_interest
            from _data_base_ as o
        left join security_table as s
            on o.secid = s.secid and o.date = s.date
        where o.secid = _secid_
            and o.open_interest > 0
            and o.best_bid > 0
            and o.best_offer - o.best_bid > 0
            and o.ss_flag = '0'
            and o.delta is not null
            and o.impl_volatility is not null
            and o.exdate - o.date <= 365
            and o.exdate - o.date > 0
            and o.date >= _start_date_
            and o.date <= _end_date_
        order by o.exdate, o.strike_price
	"""

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

def main(argv = None):

	# User specified properties:
    parser = OptionParser()

    def get_comma_separated_args(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    parser.add_option('-s','--secidlist', type='string', action='callback',
                      callback=get_comma_separated_args,
                      dest='secid_list',
                      help = """List of SECID to download option data on (e.g. 110012,106445,101966)""")
    parser.add_option("-o", "--output", action="store",
		type="string", dest="output_file",
		help = "Specify output appendix that will be added to opt_data_<appendix>.csv and dist_data_<appendix>.csv")
    parser.add_option("-b", "--startyear",
		type="int", dest="start_year",
		help = "Download data starting from this year")
    parser.add_option("-e", "--endyear",
		type="int", dest="end_year",
		help = "Download data starting up to this year")

    (options, args) = parser.parse_args()
    secid_list = [str(x) for x in options.secid_list]
    index_to_save = options.output_file
    year_start = int(options.start_year)
    year_end = int(options.end_year)

    # Connecting to WRDS server
    conn = getWRDSConnection()

    # Run a script for each year and get all data for a given company:
    year_list = list(range(year_start, year_end + 1, 1))

    df_prices = pd.DataFrame({"secid": [], "date": [], "exdate": [], "cp_flag": [],
                              "strike_price":[],"impl_volatility":[],"mid_price":[],
                              "under_price":[]})
    num_secid = len(secid_list)
    num_years = len(year_list)

    i_secid = 0

    df_list = []
    print("")
    print("--- Start loading option data ---")
    print("")
    start = time.time()
    for secid in secid_list:
        i_secid += 1
        i_year = 0
        secid = str(secid)

        for year in year_list:
            i_year += 1
            print("Secid %s, %s/%s. Year %s, %s/%s" % (str(secid), str(i_secid),
                                                       str(num_secid), str(year),
                                                       str(i_year), str(num_years)))

            start_date = "'" + str(year) + "-01-01'"
            end_date = "'" + str(year) + "-12-31'"
            data_base = "OPTIONM.OPPRCD" + str(year)

            query = getOptionMetricsQuery()

            query = query.replace('\n', ' ').replace('\t', ' ')
            query = query.replace('_start_date_', start_date).replace('_end_date_', end_date)
            query = query.replace('_secid_', secid)
            query = query.replace('_data_base_', data_base)

            df_option_i = pd.read_sql_query(query, conn)
            df_list.append(df_option_i)

    df_prices = pd.concat(df_list, ignore_index=True)

    end = time.time()
    print("")
    print("--- Time to load option data ---")
    print(end - start)
    print("")

    start = time.time()
    print("")
    print("--- Saving option data ---")
    print("")
    path_to_save_data = f"option_data_update/opt_data_{index_to_save}.csv"
    df_prices.to_csv(path_to_save_data, index = False)

    end = time.time()
    print("")
    print("--- Time to save option data ---")
    print(end - start)
    print("")

    print("")
    print("--- Start Loading Distributions Data ---")
    print("")
    # Loading data on dividend (and other) distributions:
    sec_list = "(" + ", ".join(secid_list) + ")"

    query = """
        select
            secid, payment_date, amount, distr_type
        from OPTIONM.DISTRD
        where secid in _secid_list_
        and currency = 'USD'
    """

    query = query.replace('\n', ' ').replace('\t', ' ')
    query = query.replace('_secid_list_', sec_list)
    dist_data = pd.read_sql_query(query, conn)

    conn.close()

    # Leaving only certain distribution types: '1', '3', '4', '5', '%'
    dist_data = dist_data[dist_data["distr_type"].isin(["1", "3", "4", "5", "%"])]

    print("")
    print("--- Saving Distributions Data ---")
    print("")
    dist_data.to_csv(f"option_data_update/dist_data_{index_to_save}.csv", index = False)

    print("")
    print("--- Done! ---")
    print("")


if __name__ == "__main__":
	sys.exit(main(sys.argv))
