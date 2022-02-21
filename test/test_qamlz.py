import qamlz

def test_assert():
    assert 2 == 2

def test_qamlz_TrainEnv():
    assert qamlz.TrainEnv == 'Hola, amigo!'

def test_qamlz_ModelConfig():
    assert qamlz.ModelConfig == 'Woah there, muchacho.'

def test_qamlz_Model():
    assert qamlz.Model == 'Woah there, muchacho.'