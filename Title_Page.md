# Generative AI with LangChain

## Second Edition

_Build production-ready LLM applications and advanced agents using Python, LangChain, and LangGraph_

**Ben Auffarth**  
**Leonid Kuligin**

![Packt Logo](Images/New_Packt_Logo.png)

---

## Copyright Information

**Generative AI with LangChain**  
Second Edition  
Copyright © 2025 Packt Publishing

_All rights reserved_. No part of this book may be reproduced, stored in a retrieval system, or transmitted in any form or by any means, without the prior written permission of the publisher, except in the case of brief quotations embedded in critical articles or reviews.

Every effort has been made in the preparation of this book to ensure the accuracy of the information presented. However, the information contained in this book is sold without warranty, either express or implied. Neither the authors, nor Packt Publishing or its dealers and distributors, will be held liable for any damages caused or alleged to have been caused directly or indirectly by this book.

Packt Publishing has endeavored to provide trademark information about all of the companies and products mentioned in this book by the appropriate use of capitals. However, Packt Publishing cannot guarantee the accuracy of this information.

## Publishing Information

**Portfolio Director:** Gebin George  
**Relationship Lead:** Ali Abidi  
**Project Manager:** Prajakta Naik  
**Content Engineer:** Tanya D'cruz  
**Technical Editor:** Irfa Ansari  
**Copy Editor:** Safis Editing  
**Indexer:** Manju Arasan  
**Proofreader:** Tanya D'cruz  
**Production Designer:** Ajay Patule  
**Growth Lead:** Nimisha Dua

First published: December 2023  
Second edition: May 2025  
Production reference: 2140725  
Published by Packt Publishing Ltd.  
Grosvenor House  
11 St Paul's Square  
Birmingham  
B3 1RB, UK.  
ISBN 978-1-83702-201-4

[www.packtpub.com](https://www.packtpub.com)

---

## Dedication

_To the mentors who guided me throughout my life—especially Tony Lindeberg, whose personal integrity and perseverance are a tremendous source of inspiration—and to my son, Nicholas, and my partner, Diane._

—Ben Auffarth

_To my wife, Ksenia, whose unwavering love and optimism have been my constant support over all these years; to my mother-in-law, Tatyana, whose belief in me—even in my craziest endeavors—has been an incredible source of strength; and to my kids, Matvey and Milena: I hope you'll read it one day._

—Leonid Kuligin

---

## Contributors

### About the authors

**Ben Auffarth, PhD**, is an AI implementation expert with more than 15 years of work experience. As the founder of Chelsea AI Ventures, he specializes in helping small and medium enterprises implement enterprise-grade AI solutions that deliver tangible ROI. His systems have prevented millions in fraud losses and process transactions at sub-300ms latency. With a background in computational neuroscience, Ben brings rare depth to practical AI applications—from supercomputing brain models to production systems that combine technical excellence with business strategy.

First and foremost, I want to thank my co-author, Leo—a superstar coder—who's been patient throughout and always ready when advice was needed. This book also wouldn't be what it is without the people at Packt, especially Tanya, our editor, who offered sparks of insight and encouraging words whenever needed. Finally, the reviewers were very helpful and generous with their critiques, making sure we didn't miss anything. Any errors or oversights that remain are entirely mine.

**Leonid Kuligin** is a staff AI engineer at Google Cloud, working on generative AI and classical machine learning solutions, such as demand forecasting and optimization problems. Leonid is one of the key maintainers of Google Cloud integrations on LangChain and a visiting lecturer at CDTM (a joint institution of TUM and LMU). Prior to Google, Leonid gained more than 20 years of experience building B2C and B2B applications based on complex machine learning and data processing solutions—such as search, maps, and investment management—in German, Russian, and U.S. technology, financial, and retail companies.

I want to express my sincere gratitude to all my colleagues at Google with whom I had the pleasure and joy of working, and who supported me during the creation of this book and many other endeavors. Special thanks go to Max Tschochohei, Lucio Floretta, and Thomas Cliett. My appreciation also goes to the entire LangChain community, especially Harrison Chase, whose continuous development of the LangChain framework made my work as an engineer significantly easier.

### About the reviewers

**Max Tschochohei** advises enterprise customers on how to realize their AI and ML ambitions on Google Cloud. As an engineering manager in Google Cloud Consulting, he leads teams of AI engineers on mission-critical customer projects. While his work spans the full range of AI products and solutions in the Google Cloud portfolio, he is particularly interested in agentic systems, machine learning operations, and healthcare applications of AI. Before joining Google in Munich, Max spent several years as a consultant, first with KPMG and later with the Boston Consulting Group. He also led the digital transformation of NTUC Enterprise, a Singapore government organization. Max holds a _PhD in Economics_ from Coventry University.

**Rany ElHousieny** is an AI Solutions Architect and AI Engineering Manager with over two decades of experience in AI, NLP, and ML. Throughout his career, he has focused on the development and deployment of AI models, authoring multiple articles on AI systems architecture and ethical AI deployment. He has led groundbreaking projects at companies like Microsoft, where he spearheaded advancements in NLP and the Language Understanding Intelligent Service (LUIS). Currently, he plays a pivotal role at Clearwater Analytics, driving innovation in generative AI and AI-driven financial and investment management solutions.

**Nicolas Bievre** is a Machine Learning Engineer at Meta with extensive experience in AI, recommender systems, LLMs, and generative AI, applied to advertising and healthcare. He has held key AI leadership roles at Meta and PayPal, designing and implementing large-scale recommender systems used to personalize content for hundreds of millions of users. He graduated from Stanford University, where he published peer-reviewed research in leading AI and bioinformatics journals. Internationally recognized for his contributions, Nicolas has received awards such as the "Core Ads Growth Privacy" Award and the "Outre-Mer Outstanding Talent" Award. He also serves as an AI consultant to the French government and as a reviewer for top AI organizations.

### Join our communities on Discord and Reddit

Have questions about the book or want to contribute to discussions on Generative AI and LLMs? Join our Discord server at [https://packt.link/4Bbd9](https://packt.link/4Bbd9) and our Reddit channel at [https://packt.link/wcYOQ](https://packt.link/wcYOQ) to connect, share, and collaborate with like-minded AI professionals.

| Discord QR                                  | Reddit QR                              |
| ------------------------------------------- | -------------------------------------- |
| ![Discord QR](Images/Discord_QR_Code_2.png) | ![Reddit QR](Images/Reddit_QRcode.png) |
