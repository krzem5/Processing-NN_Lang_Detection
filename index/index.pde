import java.io.File;



static String[] ALPHABET="abcdefghijklmnopqrstuvwxyz".split("");  // The alphabet
static int MODE=1;                                                // 0 -> Predicting, 1 -> Training, 2 -> Test all words
static String WORD_IN="hi";                                       // Word to be prediced (Only in MODE 1)
static int LETTERS=10;                                            // Max letters
static String[] LANGS={"english", "polish"};                      // Laungauages used in the NN



String[][] LANGS_DICTS;
NeuralNetwork nn;



void setup() {
  LANGS_DICTS=load_langs();
  load_nn();
  run();
}
String[][] load_langs() {
  String[][] data = new String[LANGS.length][];
  int i=0;
  for (String l : LANGS) {
    String[] a=loadStrings("./lang/"+l+".txt");
    String[] b=new String[a.length-1];
    for (int j=0, k=0; j<a.length; j++) {
      if (a[j].length()<=LETTERS) {
        b[k++]=a[j].toLowerCase();
      } else {
        //println("REMOVED > "+l.toUpperCase()+" > "+a[j].toLowerCase());
      }
    }
    data[i]=b;
    i++;
  }
  return data;
}
void load_nn() {
  File f = new File(dataPath("nn.json"));
  if (f.exists()&&f.isFile()) {
    nn=new NeuralNetwork(loadJSONObject("./nn.json"));
  } else {
    nn=new NeuralNetwork(LETTERS*ALPHABET.length, LETTERS, LANGS.length, 0.001);
  }
  saveJSONObject(nn.toJSON(), "./data/nn.json");
}



void run() {
  if (MODE==0) {
    String l=to_lang(nn.predict(to_float(WORD_IN)));
    println("PREDICTING > "+WORD_IN+" is "+l.toUpperCase()+"(?)");
  }
  if (MODE==1) {
    test();
    int total=0;
    for (int i=0; i<LANGS_DICTS.length; i++) {
      for (int j=0; j<LANGS_DICTS[i].length; j++) {
        if (LANGS_DICTS[i][j]!=null) {
          total++;
        }
      }
    }
    float[][]ins=new float[total][];
    float[][]outs=new float[total][];
    int idx=0;
    for (int i=0; i<LANGS_DICTS.length; i++) {
      for (int j=0; j<LANGS_DICTS[i].length; j++) {
        if (LANGS_DICTS[i][j]!=null) {
          ins[idx]=to_float(LANGS_DICTS[i][j]);
          float[] o=new float[LANGS.length];
          o[i]=1;
          outs[idx]=o;
          idx++;
        }
      }
    }
    int t=millis();
    nn.train_multiple(ins, outs, 1000 , false);
    t=millis()-t;
    println(t);
    test();
  }
  if (MODE==2) {
    test();
  }
  saveJSONObject(nn.toJSON(), "./data/nn.json");
}



float[] to_float(String s) {
  float[] data=new float[LETTERS*ALPHABET.length];
  for (int i=0; i<s.length(); i++) {
    for (int j=0; j<ALPHABET.length; j++) {
      if (ALPHABET[j].equals(str(s.charAt(i)))) {
        data[i*ALPHABET.length+j]=1;
        break;
      }
    }
  }
  return data;
}
String to_lang(float[] data) {
  String s=LANGS[0];
  float b=data[0];
  for (int i=0; i<data.length; i++) {
    if (data[i]>b) {
      b=data[i];
      s=LANGS[i];
    }
  }
  return s;
}



void test() {
  int correct=0;
  int  total=0;
  for (int i=0; i<LANGS_DICTS.length; i++) {
    for (int j=0; j<LANGS_DICTS[i].length; j++) {
      if (LANGS_DICTS[i][j]!=null) {
        String l=to_lang(nn.predict(to_float(LANGS_DICTS[i][j])));
        if (l.equals(LANGS[i])) {
          correct++;
        }
        total++;
      }
    }
  }
  println("TEST > "+(float)correct/total*100+"% accurate");
}
