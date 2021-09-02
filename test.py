from mysql import connector
import mysql.connector
from mysql.connector import errorcode
import psycopg2


conn = psycopg2.connect(
    host="ec2-44-194-225-27.compute-1.amazonaws.com",
    database="deeu0ul4vjsdi4",
    port = "5432",
    user ="qkwikqxuzcsmfu",
    password="8888ead936d03fabd6e7b60da58d0a7686c31592397689ef5d1bb0bc7c9644ca"
)
cur = conn.cursor()
cur.execute("SELECT * FROM bx_books ORDER BY id ASC LIMIT 100")
record = cur.fetchall()
print(record)