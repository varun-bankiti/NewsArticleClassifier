

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Set;
import java.io.InputStreamReader;

public class Runner {
	public static void main(String[] args) {
		String learnPoliticalSrc = "corpus/poltics_train";// args[0];
		String learnSportsSrc = "corpus/sports_train";// args[1];
		String learnBuisnessSrc = "corpus/buisness_train";// args[1];
		String testPoliticalSrcFile = "corpus/politics_test";// args[2];
		String testSportsSrcFile = "corpus/sports_test";// args[3];
		String testBuisnessSrcFile = "corpus/buisness_test";

		File testPoliticalSrc = new File(testPoliticalSrcFile);
		File testSportsSrc = new File(testSportsSrcFile);
		File testBuisnessSrc = new File(testBuisnessSrcFile);
		File politicalFolder = new File(learnPoliticalSrc);
		File sportsFolder = new File(learnSportsSrc);
		File buisnessFolder = new File(learnBuisnessSrc);
		try {
		    System.out.println("\nLearning Semantic Features...\n");
			NLPClassifier.learn();
		    System.out.println("Learning Bi-Grams.... Done\n");
			NLP_Bigram.nlp_bigram();
		    System.out.println("Learning Headwords...\n");
			ClassifierHeadWords.learn();
		    System.out.println("Learning POS Tags...\n");			
			ClassifierPOS.learn();
			if(args.length>0){			
                test(args[0]);		
			}
			else{
    		    System.out.println("Testing Political category accuracy...\n");
			    test(testPoliticalSrc,NLPClassifier.DOCUMENT_CLASS.POLITICAL);
    		    System.out.println("Testing Business category accuracy...\n");			    
			    test(testBuisnessSrc,NLPClassifier.DOCUMENT_CLASS.BUSINESS);
    		    System.out.println("Testing Sports category accuracy...\n");			    
			    test(testSportsSrc,NLPClassifier.DOCUMENT_CLASS.SPORTS);
		    }
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
    public static void test(String fileName)throws IOException{
        File newFile=new File(fileName);
	    HashMap<String, Integer> docVocab=Runner.getTokens(newFile);

	    HashMap<NLPClassifier.DOCUMENT_CLASS, Double> nbScoreMap = NLPClassifier
			    .getDocumentClass(docVocab, true);

	    HashMap<NLPClassifier.DOCUMENT_CLASS, Double> synonymsScoreMap = Synonyms
			    .getDocumentClass(docVocab, true);
	    HashMap<NLPClassifier.DOCUMENT_CLASS, Double> hyponymsScoreMap = Hyponyms
			    .getDocumentClass(docVocab, true);
	    HashMap<NLPClassifier.DOCUMENT_CLASS, Double> hypernymsScoreMap = Hypernyms
			    .getDocumentClass(docVocab, true);
	    HashMap<NLPClassifier.DOCUMENT_CLASS, Double> meronymsScoreMap = Meronyms
			    .getDocumentClass(docVocab, true);
	    HashMap<String, Integer> bigramVocab=NLP_Bigram.getTokens(newFile);
	    HashMap<NLPClassifier.DOCUMENT_CLASS, Double> bigramScoreMap = NLP_Bigram
			    .getDocumentClass(bigramVocab, true);
	    HashMap<String, Integer> headWordVocab=ClassifierHeadWords.getTokens(newFile);
	    HashMap<NLPClassifier.DOCUMENT_CLASS, Double> headwordsScoreMap = ClassifierHeadWords
			    .getDocumentClass(headWordVocab, true);
	    HashMap<String, Integer> posTagVocab=ClassifierPOS.getTokens(newFile);
	    HashMap<NLPClassifier.DOCUMENT_CLASS, Double> posTagScoreMap = ClassifierPOS
			    .getDocumentClass(posTagVocab, true);

	    Double politicalScore = nbScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
			    + synonymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
			    + hyponymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
			    + hypernymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
			    + meronymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
			    + bigramScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
			    + headwordsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
			    + posTagScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL);
	
	

	    Double sportsScore = nbScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
			    + synonymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
			    + hyponymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
			    + hypernymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
			    + meronymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
			    + bigramScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
			    + headwordsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
			    + posTagScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS);

	    Double buisnessScore = nbScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
			    + synonymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
			    + hyponymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
			    + hypernymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
			    + meronymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
			    + bigramScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
			    + headwordsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
			    + posTagScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS);


	        if (politicalScore > sportsScore && politicalScore > buisnessScore)
                System.out.println(fileName+" belongs to :"+NLPClassifier.DOCUMENT_CLASS.POLITICAL);
	        else if (buisnessScore > sportsScore && buisnessScore> politicalScore)
                System.out.println(fileName+" belongs to :"+NLPClassifier.DOCUMENT_CLASS.BUSINESS);
		    else if ( sportsScore >buisnessScore && sportsScore> politicalScore)
                System.out.println(fileName+" belongs to :"+NLPClassifier.DOCUMENT_CLASS.SPORTS);
    }
	public static void test(File folder, NLPClassifier.DOCUMENT_CLASS category)
			throws FileNotFoundException, IOException {
		String files[] = folder.list();
		int docCount = 0;
		int correctCount = 0;
		for (String file : files) {
			File newFile = new File(folder, file);
			docCount++;
			HashMap<String, Integer> docVocab=Runner.getTokens(newFile);

			HashMap<NLPClassifier.DOCUMENT_CLASS, Double> nbScoreMap = NLPClassifier
					.getDocumentClass(docVocab, true);

			HashMap<NLPClassifier.DOCUMENT_CLASS, Double> synonymsScoreMap = Synonyms
					.getDocumentClass(docVocab, true);
			HashMap<NLPClassifier.DOCUMENT_CLASS, Double> hyponymsScoreMap = Hyponyms
					.getDocumentClass(docVocab, true);
			HashMap<NLPClassifier.DOCUMENT_CLASS, Double> hypernymsScoreMap = Hypernyms
					.getDocumentClass(docVocab, true);
			HashMap<NLPClassifier.DOCUMENT_CLASS, Double> meronymsScoreMap = Meronyms
					.getDocumentClass(docVocab, true);
			HashMap<String, Integer> bigramVocab=NLP_Bigram.getTokens(newFile);
			HashMap<NLPClassifier.DOCUMENT_CLASS, Double> bigramScoreMap = NLP_Bigram
					.getDocumentClass(bigramVocab, true);
			HashMap<String, Integer> headWordVocab=ClassifierHeadWords.getTokens(newFile);
			HashMap<NLPClassifier.DOCUMENT_CLASS, Double> headwordsScoreMap = ClassifierHeadWords
					.getDocumentClass(headWordVocab, true);
			HashMap<String, Integer> posTagVocab=ClassifierPOS.getTokens(newFile);
			HashMap<NLPClassifier.DOCUMENT_CLASS, Double> posTagScoreMap = ClassifierPOS
					.getDocumentClass(posTagVocab, true);

			Double politicalScore = nbScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
					+ synonymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
					+ hyponymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
					+ hypernymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
					+ meronymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
					+ bigramScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
					+ headwordsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL)
					+ posTagScoreMap.get(NLPClassifier.DOCUMENT_CLASS.POLITICAL);
			
			

			Double sportsScore = nbScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
					+ synonymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
					+ hyponymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
					+ hypernymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
					+ meronymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
					+ bigramScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
					+ headwordsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS)
					+ posTagScoreMap.get(NLPClassifier.DOCUMENT_CLASS.SPORTS);

			Double buisnessScore = nbScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
					+ synonymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
					+ hyponymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
					+ hypernymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
					+ meronymsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
					+ bigramScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
					+ headwordsScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS)
					+ posTagScoreMap.get(NLPClassifier.DOCUMENT_CLASS.BUSINESS);

			
		if(category.equals(NLPClassifier.DOCUMENT_CLASS.POLITICAL)){
			if (politicalScore > sportsScore && politicalScore > buisnessScore)
				correctCount++;
		}
		else if (category.equals(NLPClassifier.DOCUMENT_CLASS.BUSINESS)){
			if (buisnessScore > sportsScore && buisnessScore> politicalScore)
				correctCount++;
		}
		else if(category.equals(NLPClassifier.DOCUMENT_CLASS.SPORTS)){
			if ( sportsScore >buisnessScore && sportsScore> politicalScore)
				correctCount++;			
		}
			//System.out.println("Political:"+politicalScore);
			//System.out.println("Sports:"+sportsScore);
			//System.out.println("Business:"+buisnessScore);

		}
		System.out.println(category+" category accuracy=" + correctCount
				/ Double.valueOf(docCount) * 100);

	}
	public static HashMap<String, Integer> getTokens(File newFile)throws IOException{
		BufferedReader reader = new BufferedReader(new FileReader(newFile));
		HashMap<String, Integer> docVocab = new HashMap<String, Integer>();

		String line;
		while ((line = reader.readLine()) != null) {
			line = line.replace("-", "");
			line = line.replace(",", "");
			line = line.replace(" ,", "");
			line = line.replace(", ", "");
			line = line.replace(".", "");
			line = line.replace(":", "");
			line = line.replace(" : ", "");
			line = line.replace(" / ", "");
			Set<String> existingTokens = docVocab.keySet();
			String[] tokens = line.split(NLPClassifier.splitter);
			for (int i = 0; i < tokens.length; i++) {
				if (tokens[i].equals(""))
					continue;

				if (existingTokens.contains(tokens[i])) {
					Integer count = docVocab.get(tokens[i]);
					count++;
					docVocab.put(tokens[i], count);
				} else {
					docVocab.put(tokens[i], 1);
				}

			}

		}
		return docVocab;
		
	}

}
