from nltk.stem import SnowballStemmer
import sqlite3
import pandas as pd

spanish_sn_stemmer = SnowballStemmer('spanish')

## Functions ##
def main():
    with sqlite3.connect("../../db.sqlite3") as conn:
        cur = conn.cursor()
        data = cur.execute("""
                            SELECT 
                                url,
                                description,
                                score
                            FROM 
                                news
                            WHERE 
                                preprocessed = False
                                AND DATETIME(creationDate, 'utc') >= DATETIME('now' 'localtime', 'utc', '-2 days')
                        """) \
                  .fetchall()
    
    df = pd.DataFrame(data, 
                      columns=['url', 
                               'desc', 
                               'body',
                               'score'
                               ])
    df = df[df["desc"] != ""]
    df["score"] = df["score"] + 1

    # Extract 1 random row from each 'desc' groupby
    index_one_random_from_groups = df.groupby("desc") \
                                     .sample(n=1, 
                                             weights="score") \
                                     .index
    not_index_one_random_from_groups = df.index.difference(index_one_random_from_groups)
    # Filter out duplicates
    urls_preprocessed = df.iloc[index_one_random_from_groups]["url"].tolist()
    urls_preprocessed = tuple((x, ) for x in urls_preprocessed)
    # Filter out fully processed rows
    urls_only_updated = df.iloc[not_index_one_random_from_groups]["url"].tolist()
    urls_only_updated = tuple((x, ) for x in urls_only_updated)

    # Update with True preprocessed and non-empty updateDate
    cur.executemany("""
                    UPDATE  
                        news
                    SET
                        preprocessed = True,
                        updateDate = DATETIME('now', 'localtime', 'utc')
                    WHERE
                        url = ?
                        AND DATETIME(creationDate, 'utc') >= DATETIME('now', 'localtime', 'utc', '-2 days')
                    """, urls_preprocessed)
    conn.commit()
    # Update with non-empty updateDate
    cur.executemany("""
                    UPDATE  
                        news
                    SET
                        updateDate = DATETIME('now', 'localtime', 'utc')
                    WHERE
                        url = ?
                        AND DATETIME(creationDate, 'utc') >= DATETIME('now', 'localtime', 'utc', '-2 days')
                """, urls_only_updated)

if __name__ == "__main__":
    main()