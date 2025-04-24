# This is the regex classifier we used for bootstrapping.

import re

# Very basic classifier just searching from "secur" in the text and excluding some cases that are obviously not relevant.
def regex_classifications(jsons):

    texts = [json["body"] for json in jsons]

    for i, text in enumerate(texts):

        passed = False
        results = []
        # Iterate over matches of "secur" and check its context.
        for match in re.finditer(r"secur", text.lower()):
            surronding = text[max(0, match.start() - 50): min(len(text), match.end() + 50)]
            remainder = text[match.end():]

            # We dont want it to look like a package of folder.
            if (match.start() > 0 and (text[match.start()-1] == "." or text[match.start()-1] == "/" or text[match.start()-1] == "_")):
                results.append((match.start(), match.end(), False, "ARTIFACT_START", surronding))
                continue;

            # We dont want it to look like a file with extension by checking remainder for ".extension" but also for "/"".
            if (re.search(r"^[A-Za-z0-9-_]*\.[A-Za-z0-9]+", remainder) or re.search(r"^[A-Za-z0-9-_]*\/", remainder)):
                results.append((match.start(), match.end(), False, "ARTIFACT_END", surronding))
                continue;
            
            # Scrap CamelCase
            if (re.search(r"^[a-z0-9]*[A-Z]", remainder)):
                results.append((match.start(), match.end(), False, "CAMEL_CASE", surronding))
                continue;

            results.append((match.start(), match.end(), True, "FINE", surronding))
            passed = True
        
        jsons[i]["yes"] = 1.0 if passed else 0.0
        jsons[i]["no"] = 0.0 if passed else 1.0