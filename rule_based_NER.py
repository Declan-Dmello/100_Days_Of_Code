import re
import spacy
from typing import List, Tuple, Dict


class RuleBasedNER:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        # The gazetteers
        self.organizations = {
            "microsoft", "apple", "google", "amazon", "facebook",
            "netflix", "tesla", "ibm", "intel", "adobe"
        }
        self.locations = {
            "new york", "london", "paris", "tokyo", "beijing",
            "san francisco", "los angeles", "chicago", "seattle", "boston","goa"
        }
        self.name_prefixes = {
            "mr", "mrs", "ms", "dr", "prof", "sir", "madam",
            "miss", "lord", "lady", "rev"
        }

        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\+?[\d\s-]{10,}\b')
        self.date_pattern = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')

    def identify_organizations(self, text: str) -> List[Tuple[str, int, int, str]]:
        entities = []
        doc = self.nlp(text.lower())

        # Check gazetteer
        for org in self.organizations:
            for match in re.finditer(r'\b' + re.escape(org) + r'\b', text.lower()):
                entities.append((text[match.start():match.end()],
                                 match.start(), match.end(), "ORG"))

        # Rule: Words ending in common organization suffixes
        org_suffixes = r'\b\w+\s*(Inc\.|Corp\.|Ltd\.|LLC|Company|Associates)\b'
        for match in re.finditer(org_suffixes, text):
            entities.append((text[match.start():match.end()],
                             match.start(), match.end(), "ORG"))

        return entities

    def identify_locations(self, text: str) -> List[Tuple[str, int, int, str]]:
        entities = []
        doc = self.nlp(text.lower())

        # check with the predefined list of locations
        for loc in self.locations:
            for match in re.finditer(r'\b' + re.escape(loc) + r'\b', text.lower()):
                entities.append((text[match.start():match.end()],
                                 match.start(), match.end(), "LOC"))

        # Rule -->  Words followed by location indicators
        loc_indicators = r'in|at|from|to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        for match in re.finditer(loc_indicators, text):
            if match.group(1):
                entities.append((match.group(1),
                                 match.start(1), match.end(1), "LOC"))

        return entities

    def identify_persons(self, text: str) -> List[Tuple[str, int, int, str]]:
        entities = []
        doc = self.nlp(text)

        # Rule 1: Title followed by capitalized words
        for token in doc:
            if token.text.lower() in self.name_prefixes:
                next_token = token.i + 1
                while next_token < len(doc) and doc[next_token].text[0].isupper():
                    name = doc[token.i:next_token + 1].text
                    entities.append((name,
                                     doc[token.i].idx,
                                     doc[next_token].idx + len(doc[next_token].text),
                                     "PERSON"))
                    next_token += 1

        # Rule 2: Two consecutive capitalized words
        for i in range(len(doc) - 1):
            if (doc[i].text[0].isupper() and doc[i + 1].text[0].isupper() and
                    doc[i].pos_ in ["PROPN"] and doc[i + 1].pos_ in ["PROPN"]):
                name = doc[i].text + " " + doc[i + 1].text
                entities.append((name,
                                 doc[i].idx,
                                 doc[i + 1].idx + len(doc[i + 1].text),
                                 "PERSON"))

        return entities

    def identify_dates_emails_phones(self, text: str) -> List[Tuple[str, int, int, str]]:
        entities = []

        for match in self.date_pattern.finditer(text):
            entities.append((match.group(), match.start(), match.end(), "DATE"))

        for match in self.email_pattern.finditer(text):
            entities.append((match.group(), match.start(), match.end(), "EMAIL"))

        for match in self.phone_pattern.finditer(text):
            entities.append((match.group(), match.start(), match.end(), "PHONE"))

        return entities

    def extract_entities(self, text: str) -> List[Tuple[str, int, int, str]]:
        entities = []
        entities.extend(self.identify_organizations(text))
        entities.extend(self.identify_locations(text))
        entities.extend(self.identify_persons(text))
        entities.extend(self.identify_dates_emails_phones(text))

        return sorted(entities, key=lambda x: x[1])

    def visualize_entities(self, text: str, entities: List[Tuple[str, int, int, str]]) -> str:
        result = text
        offset = 0

        for entity, start, end, label in sorted(entities, key=lambda x: x[1], reverse=True):
            highlight = f"[{entity}]({label})"
            result = result[:start + offset] + highlight + result[end + offset:]
            offset += len(highlight) - (end - start)

        return result




def main():
    ner = RuleBasedNER()

    text = """
    Mr. Judah Fernandes from Microsoft Corp. is meeting Dr. Rui Dsouza at Google's office in Goa 
    on 15/11/2024. You can reach him at rui.dsouza@microsoft.com or +1 555-123-4567.
    """

    entities = ner.extract_entities(text)

    print("\nDetailed Entities:")
    for entity, start, end, label in entities:
        print(f"{label}: {entity}")


if __name__ == "__main__":
    main()