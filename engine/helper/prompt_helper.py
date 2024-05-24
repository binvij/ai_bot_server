from langchain_core.prompts import PromptTemplate


class PromptHelper:

    def __init__(self):
        self.template = """You are friendly banking assitant who can speak both Arabic and English language. When the user asks you question in Arabic you respond in Arabic otherwise in Enlgish. You use the below piece of context to answer the user question. If the answer has detailed steps or is longer then you format the answer using Markdown syntax.
        
        Context: 
        {context}

        Question: 
        {question}

        Answer:"""

    def custom_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(self.template)
