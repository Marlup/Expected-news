import sqlite3
import pandas as pd
from constants import (
    DB_NAME_NEWS, 
)

## Functions ##
def main():
    with sqlite3.connect(DB_NAME_NEWS) as conn:
        cur = conn.cursor()
        data = cur.execute("""
                            SELECT 
                                url,
                                description,
                                score
                            FROM 
                                news
                            WHERE
                                updateDate = ''
                            AND
                                preprocessed = FALSE
                        """) \
                    .fetchall()
        print(f"Read {len(data)} rows")
        df = pd.DataFrame(data, 
                        columns=['url', 
                                'desc', 
                                'score'
                                ])
        # Filter out empty descriptions
        df_with_desc = df[df["desc"] != ""].reset_index(drop=True)
        df_with_desc["score"] = df_with_desc["score"] + 1

        # Extract 1 random row from each 'desc' groupby
        in_index_one_random_from_groups = df_with_desc.groupby("desc") \
                                            .sample(n=1, weights="score").index
        out_index_one_random_from_groups = df_with_desc.index.difference(in_index_one_random_from_groups)
        # Filter out duplicates
        urls_to_preprocess = df_with_desc.iloc[in_index_one_random_from_groups]["url"].tolist()
        urls_to_preprocess = tuple((x, ) for x in urls_to_preprocess)
        # Filter out fully processed rows
        urls_to_only_update = df_with_desc.iloc[out_index_one_random_from_groups]["url"].tolist() + df.loc[df["desc"] == "", "url"].tolist()
        urls_to_only_update = tuple((x, ) for x in urls_to_only_update)

        cur.executemany("""
            UPDATE  
                news
            SET
                preprocessed = TRUE,
                updateDate = DATETIME('now', 'localtime', 'utc')
            WHERE
                url = ?
        """, urls_to_preprocess)
        conn.commit()
        cur.executemany("""
            UPDATE  
                news
            SET
                updateDate = DATETIME('now', 'localtime', 'utc')
            WHERE
                url = ?
        """, urls_to_only_update)
        conn.commit()
        print(f"Rows processed: {len(urls_to_preprocess)}")
        print(f"Rows discarded (updateDate not empty but preprocessed is False): {len(urls_to_only_update)}")

if __name__ == "__main__":
    main()