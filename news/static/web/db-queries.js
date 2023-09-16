const sqlite3 = require('sqlite3').verbose();
const cursor = new sqlite3.Database('../../.db');

cursor.all('SELECT * FROM users', (err, rows) => {
    if (err) {
        throw err;
    }
})