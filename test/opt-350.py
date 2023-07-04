from transformers import pipeline

generator = pipeline('text-generation', model="facebook/opt-350m")
a = generator("Hello, I'm am conscious and", max_length=100)
print(a)







