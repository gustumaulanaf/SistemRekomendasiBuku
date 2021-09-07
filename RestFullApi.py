import collections
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import operator
from flask import Flask, jsonify,request
from flask_restful import Resource, Api, reqparse
import flask_sqlalchemy as sql
import pandas as pd
import matplotlib.pyplot as plt
import base64
import psycopg2, json
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
api = Api(app)

# conn = psycopg2.connect(
#     host="ec2-44-194-225-27.compute-1.amazonaws.com",
#     database="deeu0ul4vjsdi4",
#     port = "5432",
#     user ="qkwikqxuzcsmfu",
#     password="8888ead936d03fabd6e7b60da58d0a7686c31592397689ef5d1bb0bc7c9644ca"
# )
userid =0
conn = psycopg2.connect(
    host="localhost",
    database="sistem_rekomendasi_buku",
    port = "5432",
    user ="postgres",
    password=""
)
cur = conn.cursor()

#memanggil tabel dari postgresql
buku = pd.read_sql('SELECT * FROM bx_books a ORDER BY a.id ASC', con=conn)
ratings= pd.read_sql('SELECT * FROM bx_ratings a ORDER BY a.id ASC', con=conn)
users =pd.read_sql('SELECT * FROM bx_users a WHERE location LIKE %s ORDER BY a.id ASC', con=conn,params=("%"+"usa"+"%",))


#diagram distribusi rating pada tabel bx_ratings
plt.rc("font", size=15)
ratings['book_rating'].value_counts(sort=False).plot(kind='bar')
plt.title('Distribusi Rating\n')
plt.xlabel('Rating')
plt.ylabel('Jumlah')
plt.savefig('rating.png', bbox_inches='tight')
with open("rating.png", "rb") as img_file:
    rating_string = base64.b64encode(img_file.read())

#diagram distribusi usia pada tabel bx_users
plt.rc("font", size=15)
users['age'].value_counts(sort=False).plot(kind='bar')
plt.title('Distribusi Usia\n')
plt.xlabel('Usia')
plt.ylabel('Jumlah')
plt.savefig('usia.png', bbox_inches='tight')
with open("usia.png", "rb") as img_file:
    age_string = base64.b64encode(img_file.read())
# buku yang direkomendasikan
# membuat api untuk digunakan pada android
class api(Resource):
    @app.route("/login", methods = ['GET','POST'])
    def login():
        userID = request.args.get('user_id')
        password = request.args.get('password')
        query = "SELECT * FROM bx_users WHERE user_id=%s AND password=%s"
        cur.execute(query,(userID,password))
        records = cur.fetchall() 
        object_list=[]
        for row in records :
            d = collections.OrderedDict()
            d["id"] = row[0]
            d["user_id"] = row[1]
            d["location"] = row[2]
            d["age"] = row[3]
            d["avatar"] = row[4]
            d["name"] = row[6]
            object_list.append(d)
        j = json.dumps(object_list)     
        if len(object_list)>0:
            return j
        else :
            data = {'status':'Login Gagal'}
            return jsonify(data)
    @app.route("/update_profile",methods=['POST'])
    def update_profile():
        data = req.json
        return jsonify(data) 
    @app.route("/books" , methods=['GET', 'POST'])
    def books():    
        book_name = ""  
        book_name = request.args.get('book_name')
        book_sql = "SELECT * FROM bx_books WHERE book_title LIKE %(like)s ORDER BY id ASC LIMIT 100"
        cur = conn.cursor()
        cur.execute(book_sql,dict(like = '%'+book_name+'%'))
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
    
    @app.route("/recommendation",methods=['GET','POST'])
    def recommendation():
        negara = request.args.get('negara')  
        userID = request.args.get('usia')
        # Untuk memastikan signifikansi statistik,
        # pengguna dengan rating kurang dari 1,
        # dan buku dengan rating kurang dari 1 dikecualikan.
        buku = pd.read_sql('SELECT * FROM bx_books a ORDER BY a.id ASC', con=conn)
        query_ratings= "SELECT a.id, a.user_id,a.isbn, a.book_rating FROM bx_ratings a JOIN bx_users b ON a.user_id=b.user_id WHERE b.location LIKE (%(negara)s)"
        queryParams = {'negara': '%'+negara+'%'}
        ratings = pd.read_sql(sql = query_ratings, con=conn,params = queryParams)
        users =pd.read_sql('SELECT * FROM bx_users a ORDER BY a.id ASC', con=conn)
        counts1 = ratings['user_id'].value_counts()
        ratings = ratings[ratings['user_id'].isin(counts1[counts1 >= 1].index)]
        counts = ratings['book_rating'].value_counts()
        ratings = ratings[ratings['book_rating'].isin(counts[counts >= 1].index)]

        #proses collaborative filtering menggunakan knn dan cosine similarity
        #melakukan merging tabel ratings dan books berdasarkan isbn
        combine_book_rating = pd.merge(ratings, buku, on='isbn')
        columns = ['year_of_publication', 'publisher', 'book_author', 'image_url_s', 'image_url_m', 'image_url_l']
        combine_book_rating = combine_book_rating.drop(columns, axis=1)
        #combine_book_rating.head()

        #membuat field baru untuk total rating count
        combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['book_title'])
        book_ratingCount = (combine_book_rating.
            groupby(by = ['book_title'])['book_rating'].
            count().
            reset_index().
            rename(columns = {'book_rating': 'total_rating_count'})
            [['book_title', 'total_rating_count']]
            )
        #book_ratingCount.head()

        #menggabungkan data peringkat dan data total rating untuk mencari tahu buku yang populer dan yang kurang populer
        rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'book_title', right_on = 'book_title', how = 'left')

        #menentukan threshold untuk popularitas buku
        popularity_threshold = 1
        rating_popular_book = rating_with_totalRatingCount.query('total_rating_count >= @popularity_threshold')
        pd.set_option('display.float_format', lambda x: '%.3f' % x)

        #memfilter pengguna yang berlokasi di negara
        combined = rating_popular_book.merge(users, left_on = 'user_id', right_on = 'user_id', how = 'left')
        negara_user_rating = combined
        negara_user_rating=negara_user_rating.drop('age', axis=1)

        #implementasi KNN menggunakan cosine similarity
        negara_user_rating = negara_user_rating.drop_duplicates(['user_id', 'book_title'])
        negara_user_rating_pivot = negara_user_rating.pivot(index = 'book_title', columns = 'user_id', values = 'book_rating').fillna(0)
        negara_user_rating_pivot2 = negara_user_rating.pivot(index = 'user_id', columns = 'book_title', values = 'book_rating').fillna(0)
        negara_user_rating_matrix = csr_matrix(negara_user_rating_pivot.values,dtype=float)
        model_knn = NearestNeighbors(metric='cosine', algorithm = 'brute')
        model_knn.fit(negara_user_rating_matrix) 

        #melakukan pemilihan sampel buku 
        query_index = np.random.choice(negara_user_rating_pivot.shape[0])
        distances, indices = model_knn.kneighbors(negara_user_rating_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 10)
        negara_user_rating_pivot.iloc[query_index,:].values.reshape(1,-1)
        #buku yang dibaca
        negara_user_rating_pivot.index[query_index]
        rekomendasi_list=[]
        for i in range(0, len(distances.flatten())):
            d = collections.OrderedDict()
            d["id"] = i
            d["rekomendasi"] = negara_user_rating_pivot.index[indices.flatten()[i]]
            d["distances"] = distances.flatten()[i]
            rekomendasi_list.append(d)
        return json.dumps(rekomendasi_list)
    @app.route("/distribusi_usia",methods=['GET','POST'])
    def distribusi_usia():
        data = {'gambar':age_string}
        return jsonify(data)
    
    @app.route("/distribusi_rating",methods=['GET','POST'])
    def distribusi_rating():
        data = {'rating':rating_string}
        return jsonify(data)
    @app.route("/similarity",methods=['GET','POST'])
    def similarity():
        counts1 = ratings['user_id'].value_counts()
        ratings = ratings[ratings['user_id'].isin(counts1[counts1 >= 1].index)]
        counts = ratings['book_rating'].value_counts()
        ratings = ratings[ratings['book_rating'].isin(counts[counts >= 1].index)]

        #proses collaborative filtering menggunakan knn dan cosine similarity
        #melakukan merging tabel ratings dan books berdasarkan isbn
        combine_book_rating = pd.merge(ratings, buku, on='isbn')
        columns = ['year_of_publication', 'publisher', 'book_author', 'image_url_s', 'image_url_m', 'image_url_l']
        combine_book_rating = combine_book_rating.drop(columns, axis=1)
        #combine_book_rating.head()
        # #membuat field baru untuk total rating count
        combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['book_title'])
        book_ratingCount = (combine_book_rating.groupby(by = ['book_title'])['book_rating'].count().reset_index().rename(columns = {'book_rating': 'total_rating_count'})[['book_title', 'total_rating_count']])
        #book_ratingCount.head()
        #menggabungkan data peringkat dan data total rating untuk mencari tahu buku yang populer dan yang kurang populer
        rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'book_title', right_on = 'book_title', how = 'left')
        #menentukan threshold untuk popularitas buku
        popularity_threshold = 1
        rating_popular_book = rating_with_totalRatingCount.query('total_rating_count >= @popularity_threshold')
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        #memfilter pengguna yang berlokasi di negara
        combined = rating_popular_book.merge(users, left_on = 'user_id', right_on = 'user_id', how = 'left')
        negara_user_rating = combined
        negara_user_rating=negara_user_rating.drop('age', axis=1)
        #implementasi KNN menggunakan cosine similarity
        negara_user_rating = negara_user_rating.drop_duplicates(['user_id', 'book_title'])
        negara_user_rating_pivot = negara_user_rating.pivot(index = 'book_title', columns = 'user_id', values = 'book_rating').fillna(0)
        negara_user_rating_pivot2 = negara_user_rating.pivot(index = 'user_id', columns = 'book_title', values = 'book_rating').fillna(0)
        negara_user_rating_matrix = csr_matrix(negara_user_rating_pivot.values,dtype=float)
        user_id =3363 
        k=2
        user = negara_user_rating_pivot2[negara_user_rating_pivot2.index == user_id]
        other_users = negara_user_rating_pivot2[negara_user_rating_pivot2.index != user_id]

        similarities = cosine_similarity(user,other_users)[0].tolist()    
        indices = other_users.index.tolist()
            
        index_similarity = dict(zip(indices, similarities))
            
        # sort by similarity
        index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
        index_similarity_sorted.reverse()
        top_users_similarities = index_similarity_sorted[:k]
        users = [u[0] for u in top_users_similarities]
        scores = [u[1] for u in top_users_similarities]
        similarity = users,scores
    
        return jsonify(similarity)    
        
if __name__ == '__main__':
    app.run(debug=True)