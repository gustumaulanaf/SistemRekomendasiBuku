from flask import Flask, jsonify,request
from flask_restful import Resource, Api, reqparse
import flask_sqlalchemy as sql
import pandas as pd
import matplotlib.pyplot as plt

BOOKS = "BX-Books.csv"
data = pd.read_csv(BOOKS, sep=';', error_bad_lines=False,encoding="latin-1")
app = Flask(__name__)
api = Api(app)
data_arg = reqparse.RequestParser()
data_arg.add_argument("ISBN", type=str, help="Enter ISBN")
data_arg.add_argument("Book-Title", type=str, help="Enter Book-Title")
data_arg.add_argument("Book-Author", type=str, help="Enter Book-Author")
data_arg.add_argument("Year-Of-Publication", type=str, help="Enter Year-Of-Publication")
data_arg.add_argument("Publisher", type=str, help="Enter Publisher")
data_arg.add_argument("Image-URL-S", type=str, help="Enter Image-URL-S")
data_arg.add_argument("Image-URL-M", type=str, help="Enter Image-URL-M")
data_arg.add_argument("Image-URL-L", type=str, help="Enter Image-URL-L")
class read_Delete(Resource):
    def __init__(self):
        # read csv file
        data.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']

    # GET request on the url will hit this function
    @app.route("/books")
    def get():
        plt.rc("font", size=15)
        ratings.bookRating.value_counts(sort=False).plot(kind='bar')
        plt.title('Rating Distribution\n')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.savefig('system1.png', bbox_inches='tight')
        plt.show()
        
        isbn = request.args.get("isbn")
        title = request.args.get("title")
        author = request.args.get("author")
        year = request.args.get("year")
        data_fount:str
        if isbn :
         data_fount=data.loc[data["ISBN"] == isbn].to_json(orient="records")
        elif title :
         data_fount=data.loc[data["Book-Title"] == title].to_json(orient="records")
        elif author : 
         data_fount=data.loc[data["Book-Author"] == author].to_json(orient="records")
        elif year :
         data_fount=data.loc[data["Year-Of-Publication"] == year].to_json(orient="records")
        else :
          data_fount=data.to_json(orient="records")

        # find data from csv based on user input
        # return data found in csv
        return plt.show()
    # Delete request on the url will hit this function
    # def delete(self,ISBN):
    #     if ((self.data['ISBN'] == ISBN).any()):
    #         # Id it present delete data from csv
    #         self.data = self.data.drop(self.data["ISBN"].loc[self.data["ISBN"] == ISBN].index)
    #         self.data.to_csv("BX-Books.csv", index=False)
    #         return jsonify({"message": 'Deleted successfully'})
    #     else:
    #         return jsonify({"message": 'Not Present'})

class Create_Update(Resource):
    def __init__(self):
        # read data from csv
        self.data = pd.read_csv(BOOKS)

    # POST request on the url will hit this function
    def post(self):
        # data parser to parse data from url
        args = data_arg.parse_args()
        # if ID is already present
        if((self.data['ISBN']==args.ID).any()):
            return jsonify({"message": 'ISBN already exist'})
        else:
            # Save data to csv
            self.data= self.data.append(args, ignore_index=True)
            self.data.to_csv(BOOKS, index=False)
            return jsonify({"message": 'Done'})

    # PUT request on the url will hit this function
    def put(self):
        args = data_arg.parse_args()
        if ((self.data['ISBN'] == args.ID).any()):
            # if ID already present Update it
            self.data=self.data.drop(self.data["ISBN"].loc[self.data["ISBN"] == args.ID].index)
            self.data = self.data.append(args, ignore_index=True)
            self.data.to_csv(BOOKS, index=False)
            return jsonify({"message": 'Updated successfully'})
        else:
            # If ID not present Save that data to csv
            self.data = self.data.append(args, ignore_index=True)
            self.data.to_csv(BOOKS, index=False)
            return jsonify({"message": 'successfully Created'})


api.add_resource(read_Delete, '/<int:isbn>')
api.add_resource(Create_Update,'/')  
if __name__ == '__main__':
    app.run(debug=True)