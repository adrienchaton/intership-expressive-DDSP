class NoteTupleSeq:
    def __init__(self):
        self.seq = []
        self.notes_on = [] # list of pitches of notes played a this given time


    def add_note(self, note):
        self.seq.append(note)


    def show(self, indexes = None):
        if indexes == None:
            a, b = 0, len(self.seq)
        else:
            (a,b) = indexes
        for i in range(a,b):
            print(self.seq[i])

    def __repr__(self) -> str:
        s = ""
        for task in self.seq:
            t = [str(elt) for elt in task]
            s+= "("+','.join(t)+")"+"\n"
        return s

    def __eq__(self, o: object) -> bool:
        if len(self.seq)!=len(o.seq):
            return False
        else:
            for i in range(len(self.seq)):
                if self.seq[i]!=o.seq[i]:
                    return False
        return True

    def save(self, filename):
        file = open(filename, "w")
        for task in self.seq:
            t = [str(elt) for elt in task]
            s = "("+','.join(t)+")"
            file.write(s+"\n")
        file.close()

    def load(self, filename):
        file = open(filename, "r")
        line = file.readline()
        while line != "":
            line = *(int(elt) for elt in line[1:-1].split(",")),
            self.seq.append(line)
            line = file.readline()
        file.close()
