import collections
from flask import Flask, jsonify,request
from flask_restful import Resource, Api, reqparse
import flask_sqlalchemy as sql
import pandas as pd
import matplotlib.pyplot as plt
import base64
import psycopg2, json

# BOOKS = "BX-Books.csv"
# data = pd.read_csv(BOOKS, sep=';', error_bad_lines=False,encoding="latin-1")
# ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
# ratings.columns = ['userID', 'ISBN', 'bookRating']
# plt.rc("font", size=15)
# ratings.bookRating.value_counts(sort=False).plot(kind='bar')
# plt.title('Rating Distribution\n')
# plt.xlabel('Rating')
# plt.ylabel('Count')
# plt.savefig('mbuh.png', bbox_inches='tight')

# with open("mbuh.png", "rb") as img_file:
#     b64_string = base64.b64encode(img_file.read())
app = Flask(__name__)
api = Api(app)

conn = psycopg2.connect(
    host="ec2-44-194-225-27.compute-1.amazonaws.com",
    database="deeu0ul4vjsdi4",
    port = "5432",
    user ="qkwikqxuzcsmfu",
    password="8888ead936d03fabd6e7b60da58d0a7686c31592397689ef5d1bb0bc7c9644ca"
)
class api(Resource):
      @app.route("/books" , methods=['GET', 'POST'])
      def books():    
        book_name = request.args.get('book_name')
        sql = "SELECT * FROM bx_books WHERE book_title LIKE %(like)s ORDER BY id ASC LIMIT 100"
        cur = conn.cursor()
        cur.execute(sql,dict(like = '%'+book_name+'%'))
        records = cur.fetchall() 
        object_list=[]
        for row in records :
            d = collections.OrderedDict()
            d["id"] = row[0]
            d["isbn"] = row[1]
            d["book_title"] = row[2]
            d["book_author"] = row[3]
            d["year_publication"] = row[4]
            d["publisher"] = row[5]
            d["image_s"] = row[6]
            d["image_m"] = row[7]
            d["image_l"] = row[8]
            object_list.append(d)
        j = json.dumps(object_list)     
        return j

if __name__ == '__main__':
    app.run(debug=True)