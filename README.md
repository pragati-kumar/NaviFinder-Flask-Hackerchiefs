## SIH 2022 Backend

### Installation Steps:

This guide assumes Python 3.6+ and pip is installed on the system

- [Python](https://www.python.org/downloads/)

#### Basic Setup

- Clone the repository
- In a terminal, at the root folder run `pip install -r requirements.txt`

#### Environment Setup

Environment variables have been hidden for security purposes

- This app requires 2 env variables
- Create a `.env` file in the root directory, and add the following:
  - MONGO_URL=&lt;Your Mongo +srv url&gt;
  - JWT_SECRET=&lt;The JWT key for signing tokens&gt;

#### Running the application

- For Windows:
  - In the cmd window, run `set FLASK_ENV=development`
- For Mac, Linux:
  - In ther terminal, run `export FLASK_ENV=development`
- In the same terminal / cmd window run `python app.py`

The flask application should be running :tada:

> For the frontend url, copy the url after `* Running on <URL>`
