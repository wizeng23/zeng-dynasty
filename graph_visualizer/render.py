import base64
from flask import Flask, render_template, url_for

app = Flask(__name__)

def image_to_data_uri(filename):
    """Return a data URI for a static image file."""
    path = app.static_folder + '/' + filename
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = "image/png" if filename.lower().endswith("png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"

@app.route("/")
def family_tree():
    relations = [
        ("Grandpa Joe", "Dad"),
        ("Grandma Mary", "Dad"),
        ("Dad", "Me"),
        ("Mom", "Me"),
    ]
    # Build dict of node -> data URI
    images = {
        "Me": image_to_data_uri("cat.png"),
        #"Dad": image_to_data_uri("dog.png"),
    }
    return render_template("family_tree.html", relations=relations, images=images)

if __name__ == "__main__":
    app.run(debug=True)
