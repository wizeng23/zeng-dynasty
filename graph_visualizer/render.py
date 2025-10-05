from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def family_tree():
    # Family relations as (parent, child)
    relations = [
        ("Grandpa Joe", "Dad"),
        ("Grandma Mary", "Dad"),
        ("Dad", "Me"),
        ("Mom", "Me"),
        ("Dad", "Sister"),
        ("Mom", "Sister"),
    ]

    # Optional images for specific nodes
    images = {
        "Me": "/static/cat.png",
        "Dad": "/static/cat.png",
        "Mom": "cat.png",
    }

    return render_template("family_tree.html", relations=relations, images=images)

if __name__ == "__main__":
    app.run(debug=True)
