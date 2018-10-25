# Natural-Language-Inference

> From September 2018, I started working as a research intern in the Alibaba Search Group. The main research field is very related to NLI. This project is to reproduce some models of current NLI and introduces these related NLI models.

## Introduction

> The Natural Language Inference models can be used in many problems in different fields. such as question-answer, text entailment inference and so on. this repository mainly focuses on text entailment inference.

> There are usually three types of data (Premise, Hypothesis and label) to complete this task. 

flow
st=>start: Start
op=>operation: Your Operation
cond=>condition: Yes or No?
e=>end
st->op->cond
cond(yes)->e
cond(no)->op
