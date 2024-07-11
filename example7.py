# after Gemini Use Transformer LIB
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")
nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer, truncation= True)

# import pathlib
import textwrap
import os 

import google.generativeai as genai # type: ignore

# from google.colab import userdata # type: ignore

# from IPython.display import display  # type: ignore
from IPython.display import Markdown  # type: ignore

def to_markdown(text):
    text = text.replace('.','*')
    return Markdown(textwrap.indent(text,'>',predicate=lambda _:True))


os.environ['GOOGLE_API_KEY']="AIzaSyB4ZkL41utR0qWX9YC9etkXh2RHh8ciMtM"


genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel('gemini-pro')
text = """
Welcome to ZQS your go-to marketing agency for comprehensive solutions that drive success. Specializing in Social Media Marketing (SMM), Search Engine Optimization (SEO), Pay-Per-Click (PPC), development, and designing, we’re more than just a marketing company.                                                                                                                       Our team of experts is passionate about crafting tailored strategies that align with your goals. Whether you need web development that aligns with the latest trends or eye-catching designs that captivate your audience, we’ve got you covered.we understand the dynamic digital landscape and tailor our strategies to suit your unique needs. Our expertise in SMM ensures your brand stands out in the social sphere, while our SEO and PPC services boost your online visibility and drive targeted traffic.
Join hands with us, where innovation meets strategy, and let’s create a digital journey that propels your business to new heights."""



# print("\n\n")
# response1 = model.generate_content("paraphrase the text: " + input_text)
# print(response1.text)


for i in range (1,4):
    input_text = text.replace('\n',' ')
    response = model.generate_content("paraphrase the text and use the simple words and whole text in paragraph as well as length of paragraph more after paraphrasing:\n (" + text +")")
    input_text = response.text
    #use another Paraphrase Algorithem
    # from nltk.tokenize import sent_tokenize
    # sentence = sent_tokenize(input_text)
    # ar1= []
    # ar2= []

    # for x in range(0,len(sentence)-1):
    #     ar1.append(nlp(sentence[x]))
    # for y in range(0,len(ar1)-1):
    #     ar2.append(ar1[y][0]["generated_text"])


    # input_text = ' '.join(ar2)
    
    #end paraphrase algorthem
    print(f"{i}: " + input_text)
    print("\n\n\n")


