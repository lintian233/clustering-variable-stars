import astrofeatures as AF


def get_features(filepath):
    return AF.AstroDataFeatures(filepath).INIT()

if __name__ == '__main__':
    
    f = get_features(r"OGLE-SMC-LPV-11911.dat")
    print(f)