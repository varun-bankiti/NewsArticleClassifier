

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class ClassifierPOS {
	public static int ClassifierPOS_totalPoliticalCount = 0;
	public static int ClassifierPOS_totalSportsCount = 0;
	public static int ClassifierPOS_totalBuisnessCount = 0;
	public static int politicalDocCount = 0;
	public static int sportsDocCount = 0;
	public static int buisnessDocCount = 0;

	public static HashMap<String, Integer> ClassifierPOS_politicalVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> ClassifierPOS_politicalProbablity = new HashMap<String, Double>();
	public static HashMap<String, Integer> ClassifierPOS_sportsVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> ClassifierPOS_sportsProbablity = new HashMap<String, Double>();
	public static HashMap<String, Integer> ClassifierPOS_buisnessVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> ClassifierPOS_buisnessProbablity = new HashMap<String, Double>();
	public static Set<String> ClassifierPOS_vocabSet = new HashSet<String>();
	public static String splitter = "\\s+";
	public static MaxentTagger tagger;
	public static TokenizerFactory<CoreLabel> ptbTokenizerFactory;

	public static void learn() {
		String learnPoliticalSrc = "corpus/poltics_train";// args[0];
		String learnSportsSrc = "corpus/sports_train";// args[1];
		String learnBuisnessSrc = "corpus/buisness_train";// args[1];
		
		File politicalFolder = new File(learnPoliticalSrc);
		File sportsFolder = new File(learnSportsSrc);
		File buisnessFolder = new File(learnBuisnessSrc);
		// Naive bayes
		tagger = new MaxentTagger("english-left3words-distsim.tagger");
		ptbTokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(),
				"untokenizable=noneDelete");
		try {
			learnFromPolictical(politicalFolder);
			learnFromSports(sportsFolder);
			learnFromBuisness(buisnessFolder);
			ClassifierPOS_vocabSet
					.addAll(ClassifierPOS_politicalVocab.keySet());
			ClassifierPOS_vocabSet.addAll(ClassifierPOS_sportsVocab.keySet());
			ClassifierPOS_vocabSet.addAll(ClassifierPOS_buisnessVocab.keySet());
			politicalProbablity();
			sportsProbablity();
			buisnessProbablity();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	

	public static void politicalProbablity() {
		Set<String> vocab = ClassifierPOS_politicalVocab.keySet();
		Integer denominator = ClassifierPOS_totalPoliticalCount
				+ ClassifierPOS_politicalVocab.keySet().size();

		for (String token : vocab) {
			Integer wordCount = ClassifierPOS_politicalVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			ClassifierPOS_politicalProbablity.put(token, probablity);
		}

	}

	public static void sportsProbablity() {
		Set<String> vocab = ClassifierPOS_sportsVocab.keySet();
		Integer denominator = ClassifierPOS_totalSportsCount
				+ ClassifierPOS_sportsVocab.keySet().size();

		for (String token : vocab) {
			Integer wordCount = ClassifierPOS_sportsVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			ClassifierPOS_sportsProbablity.put(token, probablity);
		}

	}

	public static void buisnessProbablity() {
		Set<String> vocab = ClassifierPOS_buisnessVocab.keySet();
		Integer denominator = ClassifierPOS_totalBuisnessCount
				+ ClassifierPOS_buisnessVocab.keySet().size();

		for (String token : vocab) {
			Integer wordCount = ClassifierPOS_buisnessVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			ClassifierPOS_buisnessProbablity.put(token, probablity);
		}

	}

	public static HashMap<NLPClassifier.DOCUMENT_CLASS, Double> getDocumentClass(
			HashMap<String, Integer> docTokens, boolean removeStopWord) {
		Double politicalScore;
		Double sportsScore;
		Double buisnessScore;

		Double politicalDocProbablity = politicalDocCount
				/ Double.valueOf(politicalDocCount + sportsDocCount
						+ buisnessDocCount);
		Double sportsDocProbablity = sportsDocCount
				/ Double.valueOf(politicalDocCount + sportsDocCount
						+ buisnessDocCount);
		Double buisnessDocProbablity = buisnessDocCount
				/ Double.valueOf(politicalDocCount + sportsDocCount
						+ buisnessDocCount);

		politicalScore = Math.log(politicalDocProbablity) / Math.log(2);
		sportsScore = Math.log(sportsDocProbablity) / Math.log(2);
		buisnessScore = Math.log(buisnessDocProbablity) / Math.log(2);
		for (String token : docTokens.keySet()) {
			if (removeStopWord)
				if (checkStopWord(token))
					continue;

			int tokenCount = docTokens.get(token);

			while (tokenCount > 0) {
				if (ClassifierPOS_politicalVocab.containsKey(token)) {
					Double prob = ClassifierPOS_politicalProbablity.get(token);
					politicalScore = politicalScore + Math.log(prob)
							/ Math.log(2);
				} else {
					Double x = 1 / Double
							.valueOf(ClassifierPOS_totalPoliticalCount
									+ ClassifierPOS_vocabSet.size());
					politicalScore = politicalScore + Math.log(x) / Math.log(2);
				}

				tokenCount--;
			}
		}
		for (String token : docTokens.keySet()) {

			if (removeStopWord)
				if (checkStopWord(token))
					continue;
			int tokenCount = docTokens.get(token);

			while (tokenCount > 0) {

				if (ClassifierPOS_sportsVocab.containsKey(token)) {
					sportsScore = sportsScore
							+ Math.log(ClassifierPOS_sportsProbablity
									.get(token)) / Math.log(2);
				} else {
					Double x = 1 / Double
							.valueOf(ClassifierPOS_totalSportsCount
									+ ClassifierPOS_vocabSet.size());
					sportsScore = sportsScore + Math.log(x) / Math.log(2);
				}
				tokenCount--;
			}

		}

		for (String token : docTokens.keySet()) {

			if (removeStopWord)
				if (checkStopWord(token))
					continue;
			int tokenCount = docTokens.get(token);

			while (tokenCount > 0) {

				if (ClassifierPOS_buisnessVocab.containsKey(token)) {
					buisnessScore = buisnessScore
							+ Math.log(ClassifierPOS_buisnessProbablity
									.get(token)) / Math.log(2);
				} else {
					Double x = 1 / Double
							.valueOf(ClassifierPOS_totalBuisnessCount
									+ ClassifierPOS_vocabSet.size());
					buisnessScore = buisnessScore + Math.log(x) / Math.log(2);
				}
				tokenCount--;
			}

		}

		HashMap<NLPClassifier.DOCUMENT_CLASS, Double> scoreMap = new HashMap<NLPClassifier.DOCUMENT_CLASS, Double>();
		scoreMap.put(NLPClassifier.DOCUMENT_CLASS.POLITICAL, politicalScore);
		scoreMap.put(NLPClassifier.DOCUMENT_CLASS.BUSINESS, buisnessScore);
		scoreMap.put(NLPClassifier.DOCUMENT_CLASS.SPORTS, sportsScore);

		return scoreMap;

	}

	public static HashMap<String, Integer> getTokens(File newFile)throws IOException{
		HashMap<String, Integer> docVocab = new HashMap<String, Integer>();

		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(newFile), "utf-8"));
		DocumentPreprocessor documentPreprocessor = new DocumentPreprocessor(
				reader);
		documentPreprocessor.setTokenizerFactory(ptbTokenizerFactory);
		for (List<HasWord> sentence : documentPreprocessor) {
			List<TaggedWord> tSentence = tagger.tagSentence(sentence);
			for (TaggedWord word : tSentence) {
				String taggedWord = word.toString();
				String[] pair = taggedWord.split("/");
				if (!pair[0].matches(".*[a-zA-Z].*"))
					continue;
				if (docVocab.keySet().contains(taggedWord)) {
					Integer count = docVocab.get(taggedWord);
					count++;
					docVocab.put(taggedWord, count);
				} else {
					docVocab.put(taggedWord, 1);
				}
			}
		}
		return docVocab;
		
	}

	public static void learnFromPolictical(File politicalFolder)
			throws IOException {
		// list all the files in directory
		String files[] = politicalFolder.list();
		for (String file : files) {
			politicalDocCount++;
			File newFile = new File(politicalFolder, file);

			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(newFile), "utf-8"));
			DocumentPreprocessor documentPreprocessor = new DocumentPreprocessor(
					reader);
			documentPreprocessor.setTokenizerFactory(ptbTokenizerFactory);
			for (List<HasWord> sentence : documentPreprocessor) {
				List<TaggedWord> tSentence = tagger.tagSentence(sentence);
				for (TaggedWord word : tSentence) {
					String taggedWord = word.toString();
					String[] pair = taggedWord.split("/");
					if (!pair[0].matches(".*[a-zA-Z].*"))
						continue;
					ClassifierPOS_totalPoliticalCount++;
					if (ClassifierPOS_politicalVocab.keySet().contains(
							taggedWord)) {
						Integer count = ClassifierPOS_politicalVocab
								.get(taggedWord);
						count++;
						ClassifierPOS_politicalVocab.put(taggedWord, count);
					} else {
						ClassifierPOS_politicalVocab.put(taggedWord, 1);
					}
				}
			}
		}
	}

	public static void learnFromSports(File sportsFolder) throws IOException {
		// list all the files in directory
		String files[] = sportsFolder.list();
		for (String file : files) {
			sportsDocCount++;
			File newFile = new File(sportsFolder, file);
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(newFile), "utf-8"));
			DocumentPreprocessor documentPreprocessor = new DocumentPreprocessor(
					reader);
			documentPreprocessor.setTokenizerFactory(ptbTokenizerFactory);
			for (List<HasWord> sentence : documentPreprocessor) {
				List<TaggedWord> tSentence = tagger.tagSentence(sentence);
				for (TaggedWord word : tSentence) {
					String taggedWord = word.toString();
					String[] pair = taggedWord.split("/");
					if (!pair[0].matches(".*[a-zA-Z].*"))
						continue;
					ClassifierPOS_totalSportsCount++;
					if (ClassifierPOS_sportsVocab.keySet().contains(taggedWord)) {
						Integer count = ClassifierPOS_sportsVocab
								.get(taggedWord);
						count++;
						ClassifierPOS_sportsVocab.put(taggedWord, count);
					} else {
						ClassifierPOS_sportsVocab.put(taggedWord, 1);
					}
				}
			}
		}
	}

	public static void learnFromBuisness(File buisnessFolder)
			throws IOException {
		// list all the files in directory
		String files[] = buisnessFolder.list();
		for (String file : files) {
			buisnessDocCount++;
			File newFile = new File(buisnessFolder, file);
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(newFile), "utf-8"));
			DocumentPreprocessor documentPreprocessor = new DocumentPreprocessor(
					reader);
			documentPreprocessor.setTokenizerFactory(ptbTokenizerFactory);
			for (List<HasWord> sentence : documentPreprocessor) {
				List<TaggedWord> tSentence = tagger.tagSentence(sentence);
				for (TaggedWord word : tSentence) {
					String taggedWord = word.toString();
					String[] pair = taggedWord.split("/");
					if (!pair[0].matches(".*[a-zA-Z].*"))
						continue;
					ClassifierPOS_totalBuisnessCount++;
					if (ClassifierPOS_buisnessVocab.keySet().contains(
							taggedWord)) {
						Integer count = ClassifierPOS_buisnessVocab
								.get(taggedWord);
						count++;
						ClassifierPOS_buisnessVocab.put(taggedWord, count);
					} else {
						ClassifierPOS_buisnessVocab.put(taggedWord, 1);
					}
				}
			}
		}
	}

	public static boolean checkStopWord(String token) {

		String stopWord = "a about above after again against all am an and any are aren't as "
				+ "at be because been before being below between both but by can't cannot could "
				+ "couldn't did didn't do does doesn't doing don't down during each few for from "
				+ "further had hadn't has hasn't have haven't having he he'd he'll he's her here here's "
				+ "hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its "
				+ "itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours	 "
				+ "ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the "
				+ "their theirs them themselves then there there's these they they'd they'll they're they've this those through to too "
				+ "under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which "
				+ "while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves";

		String[] stopWordArray = stopWord.split(" ");
		for (int i = 0; i < stopWordArray.length; i++) {
			if (token.equals(stopWordArray[i]))
				return true;
		}

		return false;

	}
}
