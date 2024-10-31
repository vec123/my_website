// server.js
const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Enable CORS for all requests
app.use(cors());

// Parse JSON bodies
app.use(bodyParser.json());

// Set up SQLite database
const db = new sqlite3.Database('./comments.db', (err) => {
  if (err) {
    console.error('Error connecting to the database:', err);
  } else {
    console.log('Connected to SQLite database.');
    // Create comments table if it doesn't exist
    db.run(`
      CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        comment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
  }
});

// Endpoint to retrieve comments
app.get('/comments', (req, res) => {
  db.all('SELECT * FROM comments ORDER BY created_at DESC', (err, rows) => {
    if (err) {
      res.status(500).json({ error: 'Failed to retrieve comments' });
    } else {
      res.json(rows);
    }
  });
});

// Endpoint to add a new comment
app.post('/comments', (req, res) => {
  const { name, comment } = req.body;
  db.run(
    'INSERT INTO comments (name, comment) VALUES (?, ?)',
    [name, comment],
    function (err) {
      if (err) {
        res.status(500).json({ error: 'Failed to add comment' });
      } else {
        res.status(201).json({ id: this.lastID, name, comment });
      }
    }
  );
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
