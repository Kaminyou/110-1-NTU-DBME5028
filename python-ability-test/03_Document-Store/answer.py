class DocumentStore(object):

    def __init__(self, capacity): ##
        self._capacity = capacity
        self._documents = []

    @property
    def capacity(self):
        return self._capacity

    @property
    def documents(self):
        return self._documents[:] ##

    def add_document(self, document):
        if(len(self._documents) + 1 > self._capacity): ##
            raise Exception('Document store is full') 
        self._documents.append(document)
    
    def __repr__(self):
        return "Document store: " + str(len(self._documents)) + "/" + str(self._capacity) ##

#To see the output, uncomment the lines belows:
#document_store = DocumentStore(2)
#document_store.add_document("document")
#print(document_store)