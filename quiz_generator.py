import spacy
import pandas as pd
from keybert import KeyBERT
from transformers import T5Tokenizer, T5ForConditionalGeneration

nlp = spacy.load("en_core_web_sm")
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")

kw_model = KeyBERT()

def extract_answers(text):
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents]
    if not ents:
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        ents = [kw[0] for kw in keywords]
    return list(set(ents))

def generate_question(context, answer):
    if answer in context:
        highlighted = context.replace(answer, f"<hl> {answer} <hl>")
        input_text = f"generate question: {highlighted}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=64)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question
    return None

def generate_fill_in_blank(question, answer):
    return question.replace(answer, "_____")

def generate_mcq_options(answer, context):
    keywords = kw_model.extract_keywords(context, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    distractors = [kw[0] for kw in keywords if kw[0].lower() != answer.lower()]
    options = list(set([answer] + distractors[:3]))
    while len(options) < 4:
        options.append("None of the above")
    return options

def detect_difficulty(answer, question):
    if len(answer.split()) == 1 and answer.istitle():
        return "Easy"
    elif len(answer.split()) <= 3:
        return "Medium"
    else:
        return "Hard"

def generate_quiz(text):
    answers = extract_answers(text)
    quiz_data = []

    for answer in answers:
        question = generate_question(text, answer)
        if not question:
            continue
        fill_blank = generate_fill_in_blank(question, answer)
        mcq_options = generate_mcq_options(answer, text)
        difficulty = detect_difficulty(answer, question)

        quiz_data.append({
            "Question": question,
            "Answer": answer,
            "Fill-in-the-blank": fill_blank,
            "MCQ Options": mcq_options,
            "Difficulty": difficulty
        })

    return pd.DataFrame(quiz_data)
