
from flask import Flask, render_template, request, send_file
from quiz_generator import generate_quiz
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd

app = Flask(__name__)

latest_quiz_data = pd.DataFrame()

@app.route("/", methods=["GET", "POST"])
def index():
    global latest_quiz_data

    paragraph = ""
    difficulty = ""
    quiz_df = None
    questions = None

    if request.method == "POST":
        paragraph = request.form.get("paragraph", "")
        difficulty = request.form.get("difficulty", "")

        full_df = generate_quiz(paragraph)
        latest_quiz_data = full_df.copy()

        if difficulty:
            filtered_df = full_df[full_df["Difficulty"].str.lower() == difficulty.lower()]
        else:
            filtered_df = full_df

        quiz_df = filtered_df
        questions = [
            {"question": row["Question"], "answer": row["Answer"]}
            for _, row in filtered_df.iterrows()
        ]

    return render_template("index.html", quiz_df=quiz_df, questions=questions,
                           paragraph=paragraph, difficulty=difficulty)


@app.route("/download/pdf")
def download_pdf():
    global latest_quiz_data
    if latest_quiz_data.empty:
        return "No quiz data to export", 400

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    p.setFont("Helvetica", 12)
    p.drawString(50, y, "AI Generated Quiz")
    y -= 20

    for i, row in latest_quiz_data.iterrows():
        p.drawString(50, y, f"{i + 1}. {row['Question']} (Difficulty: {row['Difficulty']})")
        y -= 15
        p.drawString(70, y, f"Answer: {row['Answer']}")
        y -= 15
        p.drawString(70, y, f"Fill in the blank: {row['Fill-in-the-blank']}")
        y -= 15
        p.drawString(70, y, f"Options: {', '.join(row['MCQ Options'])}")
        y -= 25

        if y < 100:
            p.showPage()
            y = height - 40

    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name="quiz.pdf", mimetype='application/pdf')


if __name__ == "__main__":
    app.run(debug=True)
