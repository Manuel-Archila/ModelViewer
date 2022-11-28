class Obj(object):
    def __init__(self, filename):
        with open(filename) as f:
            self.lines = f.read().splitlines()
        
        self.vertices = []
        self.faces = []
        self.vt_vertices = []
        self.nvertices = []

        for line in self.lines:
            if line:
                #print(line)
                prefix, value = line.split(' ', 1)

                if prefix == 'v':
                    self.vertices.append(
                        list(
                            map(float, value.split(' '))))
                
                if prefix == 'vt':
                    self.vt_vertices.append(
                        list(
                            map(float, value.split(' '))))
                if prefix == 'vn':
                    self.nvertices.append(
                        list(
                            map(float, value.split(' '))))
                if prefix == 'f':
                    self.faces.append([
                        list(
                            map(self.doubleDiag, face.split('/')))
                                for face in value.split(' ')])

    def doubleDiag(self, n):
        if n == '':
            return 0
        else:
            return int(n)