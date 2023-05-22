import sqlite3

def insert_news_one(data):
    
    insert_one_str = f"""INSERT INTO News VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    query_output = False
    try:
        cur.execute(insert_one_str, data)
        con.commit()
        query_output = True
    except:
        print("Error. insert_news_one() method failed")
    query_output

def insert_news_many(data):
    insert_many_str = """INSERT INTO News VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    query_output = False
    try:
        cur.executemany(insert_one_str, data)
        con.commit()
        query_output = True
    except:
        print("Error. insert_news_many() method failed")
    return query_output

def select_all(data):
    select_query_str = """
        SELECT *
            FROM News;
        """
    query_output = None
    try:
        query_output = cur.execute(select_query_str, data)
        con.commit()
    except:
        print("Error. insert_news_many() method failed")
    return query_output

def select_filter_between_date(dates, order_field=None):
    if order_field is None:
        order_field = "creationDate"
    select_query_str = """
        SELECT *
            FROM News
            WHERE 
                creationDate BETWEEN ? AND ?
        ORDER BY
            ? DESC;
        """
    query_output = None
    try:
        query_output = cur.execute(select_query_str, dates + [order_field])
        con.commit()
    except:
        print("Error. select_filter_between_date() method failed")
    return query_output