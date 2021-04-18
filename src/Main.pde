import java.io.File;



final String[] ALPHABET={"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"};
final int MODE=1;
final String WORD_IN="hi";
final int LETTERS=10;
final String[] LANGS={"english","polish"};



String[][] LANGS_DICTS;
NeuralNetwork nn;



void setup(){
	LANGS_DICTS=load_langs();
	File f=new File(dataPath("nn.json"));
	if (f.exists()&&f.isFile()){
		nn=new NeuralNetwork(loadJSONObject("nn.json"));
	}
	else{
		nn=new NeuralNetwork(LETTERS*ALPHABET.length,LETTERS,LANGS.length,0.001);
	}
	saveJSONObject(nn.toJSON(),"data/nn.json");
	if (MODE==0){
		String l=to_lang(nn.predict(to_float(WORD_IN)));
		println("PREDICTING > "+WORD_IN+" is "+l.toUpperCase()+"(?)");
	}
	else if (MODE==1){
		test();
		int total=0;
		for (int i=0;i<LANGS_DICTS.length;i++){
			for (int j=0;j<LANGS_DICTS[i].length;j++){
				if (LANGS_DICTS[i][j]!=null){
					total++;
				}
			}
		}
		float[][] ins=new float[total][];
		float[][] outs=new float[total][];
		int idx=0;
		for (int i=0;i<LANGS_DICTS.length;i++){
			for (int j=0;j<LANGS_DICTS[i].length;j++){
				if (LANGS_DICTS[i][j]!=null){
					ins[idx]=to_float(LANGS_DICTS[i][j]);
					float[] o=new float[LANGS.length];
					o[i]=1;
					outs[idx]=o;
					idx++;
				}
			}
		}
		int t=millis();
		nn.train_multiple(ins,outs,1000);
		t=millis()-t;
		println(t);
		test();
	}
	else if (MODE==2){
		test();
	}
	saveJSONObject(nn.toJSON(),"data/nn.json");
}



String[][] load_langs(){
	String[][] data=new String[LANGS.length][];
	int i=0;
	for (String l:LANGS){
		String[] a=loadStrings("lang/"+l+".txt");
		String[] b=new String[a.length-1];
		for (int j=0,k=0;j<a.length;j++){
			if (a[j].length()<=LETTERS){
				b[k++]=a[j].toLowerCase();
			}
			else{
				println("REMOVED > "+l.toUpperCase()+" > "+a[j].toLowerCase());
			}
		}
		data[i]=b;
		i++;
	}
	return data;
}



float[] to_float(String s){
	float[] data=new float[LETTERS*ALPHABET.length];
	for (int i=0;i<s.length();i++){
		for (int j=0;j<ALPHABET.length;j++){
			if (ALPHABET[j].equals(str(s.charAt(i)))){
				data[i*ALPHABET.length+j]=1;
				break;
			}
		}
	}
	return data;
}



String to_lang(float[] data){
	String s=LANGS[0];
	float b=data[0];
	for (int i=0;i<data.length;i++){
		if (data[i]>b){
			b=data[i];
			s=LANGS[i];
		}
	}
	return s;
}



void test(){
	int correct=0;
	int total=0;
	for (int i=0;i<LANGS_DICTS.length;i++){
		for (int j=0;j<LANGS_DICTS[i].length;j++){
			if (LANGS_DICTS[i][j]!=null){
				String l=to_lang(nn.predict(to_float(LANGS_DICTS[i][j])));
				if (l.equals(LANGS[i])){
					correct++;
				}
				total++;
			}
		}
	}
	println("TEST > "+(float)correct/total*100+"% accurate");
}
