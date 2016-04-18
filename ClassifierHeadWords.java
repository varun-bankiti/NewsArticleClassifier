

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;

public class ClassifierHeadWords {
	public static int ClassifierHeadWords_totalPoliticalCount = 0;
	public static int ClassifierHeadWords_totalSportsCount = 0;
	public static int ClassifierHeadWords_totalBuisnessCount = 0;
	public static int politicalDocCount = 0;
	public static int sportsDocCount = 0;
	public static int buisnessDocCount = 0;

	public static HashMap<String, Integer> ClassifierHeadWords_politicalVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> ClassifierHeadWords_politicalProbablity = new HashMap<String, Double>();
	public static HashMap<String, Integer> ClassifierHeadWords_sportsVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> ClassifierHeadWords_sportsProbablity = new HashMap<String, Double>();
	public static HashMap<String, Integer> ClassifierHeadWords_buisnessVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> ClassifierHeadWords_buisnessProbablity = new HashMap<String, Double>();
	public static Set<String> ClassifierHeadWords_vocabSet = new HashSet<String>();
	public static String splitter = "\\s+";
	public static LexicalizedParser lp;

	public static void learn() {
		lp = LexicalizedParser
				.loadModel("edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz");
		String learnPoliticalSrc = "corpus/poltics_train";// args[0];
		String learnSportsSrc = "corpus/sports_train";// args[1];
		String learnBuisnessSrc = "corpus/buisness_train";// args[1];
		File politicalFolder = new File(learnPoliticalSrc);
		File sportsFolder = new File(learnSportsSrc);
		File buisnessFolder = new File(learnBuisnessSrc);
		try {
			learnFromPolictical(politicalFolder);
			learnFromSports(sportsFolder);
			learnFromBuisness(buisnessFolder);
			ClassifierHeadWords_vocabSet
					.addAll(ClassifierHeadWords_politicalVocab.keySet());
			ClassifierHeadWords_vocabSet.addAll(ClassifierHeadWords_sportsVocab
					.keySet());
			ClassifierHeadWords_vocabSet
					.addAll(ClassifierHeadWords_buisnessVocab.keySet());
			politicalProbablity();
			sportsProbablity();
			buisnessProbablity();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void politicalProbablity() {
		Integer denominator = ClassifierHeadWords_totalPoliticalCount;
		Set<String> vocab = ClassifierHeadWords_politicalVocab.keySet();
		for (String token : vocab) {
			Integer wordCount = ClassifierHeadWords_politicalVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			ClassifierHeadWords_politicalProbablity.put(token, probablity);
		}
	}

	public static void sportsProbablity() {
		Integer denominator = ClassifierHeadWords_totalSportsCount;
		Set<String> vocab = ClassifierHeadWords_sportsVocab.keySet();
		for (String token : vocab) {
			Integer wordCount = ClassifierHeadWords_sportsVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			ClassifierHeadWords_sportsProbablity.put(token, probablity);
		}

	}

	public static void buisnessProbablity() {
		Integer denominator = ClassifierHeadWords_totalBuisnessCount;
		Set<String> vocab = ClassifierHeadWords_buisnessVocab.keySet();
		for (String token : vocab) {
			Integer wordCount = ClassifierHeadWords_buisnessVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			ClassifierHeadWords_buisnessProbablity.put(token, probablity);
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
				if (ClassifierHeadWords_politicalVocab.containsKey(token)) {
					Double prob = ClassifierHeadWords_politicalProbablity
							.get(token);
					politicalScore = politicalScore + Math.log(prob)
							/ Math.log(2);
				} else {
					Double x = 1 / Double
							.valueOf(ClassifierHeadWords_totalPoliticalCount
									+ ClassifierHeadWords_vocabSet.size());
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

				if (ClassifierHeadWords_sportsVocab.containsKey(token)) {
					sportsScore = sportsScore
							+ Math.log(ClassifierHeadWords_sportsProbablity
									.get(token)) / Math.log(2);
				} else {
					Double x = 1 / Double
							.valueOf(ClassifierHeadWords_totalSportsCount
									+ ClassifierHeadWords_vocabSet.size());
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

				if (ClassifierHeadWords_buisnessVocab.containsKey(token)) {
					buisnessScore = buisnessScore
							+ Math.log(ClassifierHeadWords_buisnessProbablity
									.get(token)) / Math.log(2);
				} else {
					Double x = 1 / Double
							.valueOf(ClassifierHeadWords_totalBuisnessCount
									+ ClassifierHeadWords_vocabSet.size());
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

	public static HashMap<String, Integer> getTokens(File newFile)
			throws IOException {
		HashMap<String, Integer> docVocab = new HashMap<String, Integer>();
		Scanner read = new Scanner(newFile);
		read.useDelimiter("[.?!]");
		String sentence, headWord;
		while (read.hasNext()) {
			sentence = read.next();
			sentence = sentence.replaceAll("\\r?\\n", " ");
			if (sentence.length() > 10 && sentence.length() < 100) {
				headWord = getHeadWord(lp, sentence);
				if (headWord.length() > 0) {
					if (docVocab.keySet().contains(headWord)) {
						Integer count = docVocab.get(headWord);
						count++;
						docVocab.put(headWord, count);
					} else {
						docVocab.put(headWord, 1);
					}
				}

			}
		}
		return docVocab;
	}

	public static String getHeadWord(LexicalizedParser lp, String sent2) {
		Tree parse;
		TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(
				new CoreLabelTokenFactory(), "");
		Tokenizer<CoreLabel> tok = tokenizerFactory
				.getTokenizer(new StringReader(sent2));
		List<CoreLabel> rawWords2 = tok.tokenize();
		parse = lp.apply(rawWords2);
		TreebankLanguagePack tlp = new PennTreebankLanguagePack();
		GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
		GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
		List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
		for (int i = 0; i < tdl.size(); i++) {
			if (tdl.get(i).reln().getLongName().equals("root"))
				return tdl.get(i).dep().toString();
		}
		return "";
	}

	public static void learnFromPolictical(File politicalFolder)
			throws IOException {
		// list all the files in directory
		String files[] = politicalFolder.list();
		String headWord;
		for (String file : files) {
			File newFile = new File(politicalFolder, file);
			if (newFile.length() / 1024.0 >4)
				continue;
			politicalDocCount++;
			Scanner read = new Scanner(newFile);
			read.useDelimiter("[.?!]");
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			String sentence;
			while (read.hasNext()) {
				sentence = read.next();
				sentence = sentence.replaceAll("\\r?\\n", " ");
				if (sentence.length() > 10 && sentence.length() < 100) {
					headWord = getHeadWord(lp, sentence);
					if (headWord.length() > 0) {
						ClassifierHeadWords_totalPoliticalCount++;
						if (ClassifierHeadWords_politicalVocab.keySet()
								.contains(headWord)) {
							Integer count = ClassifierHeadWords_politicalVocab
									.get(headWord);
							count++;
							ClassifierHeadWords_politicalVocab.put(headWord,
									count);
						} else {
							ClassifierHeadWords_politicalVocab.put(headWord, 1);
						}
					}

				}
			}
		}
	}

	public static void learnFromSports(File sportsFolder) throws IOException {
		String files[] = sportsFolder.list();
		String headWord;
		for (String file : files) {
			File newFile = new File(sportsFolder, file);
			if (newFile.length() / 1024.0 > 5)
				continue;
			sportsDocCount++;
			Scanner read = new Scanner(newFile);
			read.useDelimiter("[.?!]");
			String sentence;
			while (read.hasNext()) {
				sentence = read.next();
				sentence = sentence.replaceAll("\\r?\\n", " ");
				if (sentence.length() > 10 && sentence.length() < 100) {
					headWord = getHeadWord(lp, sentence);
					if (headWord.length() > 0) {
						ClassifierHeadWords_totalSportsCount++;
						if (ClassifierHeadWords_sportsVocab.keySet().contains(
								headWord)) {
							Integer count = ClassifierHeadWords_sportsVocab
									.get(headWord);
							count++;
							ClassifierHeadWords_sportsVocab
									.put(headWord, count);
						} else {
							ClassifierHeadWords_sportsVocab.put(headWord, 1);
						}
					}

				}
			}
		}
	}

	public static void learnFromBuisness(File buisnessFolder)
			throws IOException {
		// list all the files in directory
		String files[] = buisnessFolder.list();
		String headWord;
		for (String file : files) {
			File newFile = new File(buisnessFolder, file);
			if (newFile.length() / 1024.0 > 5)
				continue;
			buisnessDocCount++;
			Scanner read = new Scanner(newFile);
			read.useDelimiter("[.?!]");
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			String sentence;
			while (read.hasNext()) {
				sentence = read.next();
				sentence = sentence.replaceAll("\\r?\\n", " ");
				if (sentence.length() > 10 && sentence.length() < 100) {
					headWord = getHeadWord(lp, sentence);
					if (headWord.length() > 0) {
						ClassifierHeadWords_totalBuisnessCount++;
						if (ClassifierHeadWords_buisnessVocab.keySet()
								.contains(headWord)) {
							Integer count = ClassifierHeadWords_buisnessVocab
									.get(headWord);
							count++;
							ClassifierHeadWords_buisnessVocab.put(headWord,
									count);
						} else {
							ClassifierHeadWords_buisnessVocab.put(headWord, 1);
						}
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
