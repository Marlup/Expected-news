SELECT creationDate AS raw_date,
       DATETIME(creationDate) AS datetime,
       DATETIME('now') AS now, -- UTC datetime
       DATETIME('now', 'localtime') AS now_local,
       DATETIME('now', 'utc')  AS now_utc,
       DATETIME('now', 'localtime', 'utc') AS now_local_utc,
       TIME(creationDate) as time,
       updateDate
  FROM db.news
  WHERE preprocessed = False