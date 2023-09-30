import sqlite3
import pandas as pd

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
                                DATETIME(updateDate) != ''
                            AND
                                preprocessed = False
                            AND 
                                DATETIME(creationDate, 'utc') >= DATETIME('now', 'localtime', 'utc', '-2 days')
                        """) \
                  .fetchall()
    
    df = pd.DataFrame(data, 
                      columns=['url', 
                               'desc', 
                               'score'
                               ])
    # Filter out empty descriptions
    df = df[df["desc"] != ""].reset_index(drop=True)
    df["score"] = df["score"] + 1

    # Extract 1 random row from each 'desc' groupby
    in_index_one_random_from_groups = df.groupby("desc") \
                                        .sample(n=1, 
                                                weights="score").index
    out_index_one_random_from_groups = df.index.difference(in_index_one_random_from_groups)
    # Filter out duplicates
    urls_to_preprocess = df.iloc[in_index_one_random_from_groups]["url"].tolist()
    urls_to_preprocess = tuple((x, ) for x in urls_to_preprocess)
    # Filter out fully processed rows
    urls_to_only_update = df.iloc[out_index_one_random_from_groups]["url"].tolist()
    urls_to_only_update = tuple((x, ) for x in urls_to_only_update)
    
    with sqlite3.connect("../../db.sqlite3") as conn:
        cur = conn.cursor()
        # Update with True preprocessed and non-empty updateDate
        cur.executemany("""
                        UPDATE 
                            news
                        SET
                            preprocessed = True,
                            updateDate = DATETIME('now', 'localtime', 'utc')
                        WHERE
                            DATETIME(updateDate) != ''
                        AND
                            DATETIME(creationDate, 'utc') >= DATETIME('now', 'localtime', 'utc', '-2 days')
                        AND
                            url = ?
                        """, urls_to_preprocess)
        conn.commit()
        print(f"{len(urls_to_preprocess)} unique rows processed and production (web-query) ready")
        # Update with non-empty updateDate
        cur.executemany("""
                        UPDATE  
                            news
                        SET
                            updateDate = DATETIME('now', 'localtime', 'utc')
                        WHERE
                            DATETIME(updateDate) != ''
                        AND
                            DATETIME(creationDate, 'utc') >= DATETIME('now', 'localtime', 'utc', '-2 days')
                        AND
                            url = ?
                        """, urls_to_only_update)
        conn.commit()
        print(f"{len(urls_to_only_update)} duplicated rows processed and unavailable for production")

if __name__ == "__main__":
    main()