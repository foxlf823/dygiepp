
class Entity:
    def __init__(self):
        self.id = None
        self.type = None

        self.spans = [] # a couple of spans, list (start, end)
        self.tkSpans = []
        self.name = []

        self.sent_idx = None
        self.norm_ids = []

    def equals(self, other):

        if self.type == other.type and len(self.spans) == len(other.spans) :

            for i in range(len(self.spans)) :

                if self.spans[i][0] != other.spans[i][0] or self.spans[i][1] != other.spans[i][1]:
                    return False

            return True
        else:
            return False

    def equals_span(self, other):
        if len(self.spans) == len(other.spans):

            for i in range(len(self.spans)):

                if self.spans[i][0] != other.spans[i][0] or self.spans[i][1] != other.spans[i][1]:
                    return False

            return True

        else:
            return False

    def equalsTkSpan(self, other):
        if len(self.tkSpans) == len(other.tkSpans):

            for i in range(len(self.tkSpans)):

                if self.tkSpans[i][0] != other.tkSpans[i][0] or self.tkSpans[i][1] != other.tkSpans[i][1]:
                    return False

            return True

        else:
            return False



class Document:
    def __init__(self):
        self.entities = None
        self.sentences = None
        self.name = None
        self.text = None


class Sentence:
    def __init__(self):
        self.tokens = None
        self.idx = None
        self.text = None
        self.start = None
        self.end = None

class Token:
    def __init__(self):
        self.text = None
        self.start = None
        self.end = None
        self.idx = None



