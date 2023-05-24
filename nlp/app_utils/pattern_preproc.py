import re

def preprocess(input):

    patterns = [r"b'",
        r'b"',
        r"b\\",
        r"\\n",
        r'[^\w\s]', 
        r'[!@#$%<.*?>]+',#Remove simbols Remove HTML tags/markups
        r"[0-9]"
        ] 
    rem_spaces = r" +"

    if not isinstance(input, list):
        input = str(input)
        if isinstance(input, str):
            docx = input
            for pattern in patterns:
                matches = re.findall(pattern, input)
                if matches:
                    for match in matches:
                        docx = docx.replace(match, '')
                    space_match = re.findall(rem_spaces, input)
                    for sm in space_match:
                        docx = docx.replace(sm, ' ')
            docx = docx.strip()
            return docx
    
    if isinstance(input, list):
        X =[]

        for doc in input:
            docx = str(doc)
            for pattern in patterns:
                matches = re.findall(pattern, str(doc))
                if matches:
                    for match in matches:
                        docx = docx.replace(match, '')
                space_match = re.findall(rem_spaces, docx)
                for sm in space_match:
                    docx = docx.replace(sm, ' ')
            docx = docx.strip()
            X.append(docx)

        return X