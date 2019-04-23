# Dependencies
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import pickle
import traceback
import pandas as pd

# Your API definition
app = Flask(__name__, static_url_path="", static_folder="static", template_folder="templates")

with open("word_vectorizer.pkl", "rb") as vec:
    word_vectorizer = pickle.load(vec)

with open("toxic_clf.pkl", "rb") as m_toxic:
    toxic_clf = pickle.load(m_toxic)

@app.route("/")
def hello():
    # return send_from_directory("static", filename="index.html")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if word_vectorizer and toxic_clf:
        try:
            comment = request.form["comment"]
            new_samples = pd.Series(comment)
            X_new = word_vectorizer.transform(new_samples)
            pred = toxic_clf.predict_proba(X_new)[:, 1]
            result = "<i>\"" + comment + "\"</i>" + "  <br><b>Toxicity Score: " + str(round(pred[0],4)) +"</b>"            
            return result

        except:

            return jsonify({"trace": traceback.format_exc()})
    else:
        print ("Train the model first")
        return ("No model here to use")

if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don"t provide any port the port will be set to 12345
    
    # serve efficiently a large model on a machine with many cores with many gunicorn workers, you can share the model parameters in memory using memory mapping


    app.run(port=port, debug=True)