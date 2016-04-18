

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;


public class NLP_Bigram {
	public static int NLP_Bigram_totalPoliticalCount = 0;
	public static int NLP_Bigram_totalSportsCount = 0;
	public static int NLP_Bigram_totalBuisnessCount = 0;
	public static int politicalDocCount = 0;
	public static int sportsDocCount = 0;
	public static int buisnessDocCount = 0;

	public static HashMap<String, Integer> NLP_Bigram_politicalVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> NLP_Bigram_politicalProbablity = new HashMap<String, Double>();
	public static HashMap<String, Integer> NLP_Bigram_sportsVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> NLP_Bigram_sportsProbablity = new HashMap<String, Double>();
	public static HashMap<String, Integer> NLP_Bigram_buisnessVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> NLP_Bigram_buisnessProbablity = new HashMap<String, Double>();
	public static Set<String> NLP_Bigram_vocabSet = new HashSet<String>();
	public static String splitter = "\\s+";

	public static void nlp_bigram() {
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
			NLP_Bigram_vocabSet.addAll(NLP_Bigram_politicalVocab.keySet());
			NLP_Bigram_vocabSet.addAll(NLP_Bigram_sportsVocab.keySet());
			NLP_Bigram_vocabSet.addAll(NLP_Bigram_buisnessVocab.keySet());
			politicalProbablity();
			sportsProbablity();
			buisnessProbablity();
			// testPolitical(testPoliticalSrc);
			// testSports(testSportsSrc);
			// testBuisness(testBuisnessSrc);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public static void politicalProbablity() {
		Set<String> vocab = NLP_Bigram_politicalVocab.keySet();
		Integer denominator = NLP_Bigram_totalPoliticalCount
				+ NLP_Bigram_politicalVocab.keySet().size();

		for (String token : vocab) {
			Integer wordCount = NLP_Bigram_politicalVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			NLP_Bigram_politicalProbablity.put(token, probablity);
		}

	}

	public static void sportsProbablity() {
		Set<String> vocab = NLP_Bigram_sportsVocab.keySet();
		Integer denominator = NLP_Bigram_totalSportsCount
				+ NLP_Bigram_sportsVocab.keySet().size();

		for (String token : vocab) {
			Integer wordCount = NLP_Bigram_sportsVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			NLP_Bigram_sportsProbablity.put(token, probablity);
		}

	}

	public static void buisnessProbablity() {
		Set<String> vocab = NLP_Bigram_buisnessVocab.keySet();
		Integer denominator = NLP_Bigram_totalBuisnessCount
				+ NLP_Bigram_buisnessVocab.keySet().size();

		for (String token : vocab) {
			Integer wordCount = NLP_Bigram_buisnessVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			NLP_Bigram_buisnessProbablity.put(token, probablity);
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
				if (NLP_Bigram_politicalVocab.containsKey(token)) {
					int politicalVocabTokenCount = docTokens.get(token);
					while (politicalVocabTokenCount != 0) {
						Double prob = NLP_Bigram_politicalProbablity.get(token);
						politicalScore = politicalScore + Math.log(prob)
								/ Math.log(2);
						politicalVocabTokenCount--;
					}
				} else {
					int politicalVocabTokenCount = docTokens.get(token);
					while (politicalVocabTokenCount != 0) {
						Double x = 1 / Double
								.valueOf(NLP_Bigram_totalPoliticalCount
										+ NLP_Bigram_vocabSet.size());
						politicalScore = politicalScore + Math.log(x)
								/ Math.log(2);
						politicalVocabTokenCount--;
					}
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

				if (NLP_Bigram_sportsVocab.containsKey(token)) {
					int sportsVocabTokenCount = docTokens.get(token);
					while (sportsVocabTokenCount != 0) {
						sportsScore = sportsScore
								+ Math.log(NLP_Bigram_sportsProbablity
										.get(token)) / Math.log(2);
						sportsVocabTokenCount--;
					}
				} else {
					int sportsVocabTokenCount = docTokens.get(token);
					while (sportsVocabTokenCount != 0) {
						Double x = 1 / Double
								.valueOf(NLP_Bigram_totalSportsCount
										+ NLP_Bigram_vocabSet.size());
						sportsScore = sportsScore + Math.log(x) / Math.log(2);
						sportsVocabTokenCount--;
					}
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

				if (NLP_Bigram_buisnessVocab.containsKey(token)) {
					int buisnessVocabTokenCount = docTokens.get(token);
					while (buisnessVocabTokenCount != 0) {
						buisnessScore = buisnessScore
								+ Math.log(NLP_Bigram_buisnessProbablity
										.get(token)) / Math.log(2);
						buisnessVocabTokenCount--;
					}
				} else {
					int buisnessVocabTokenCount = docTokens.get(token);
					while (buisnessVocabTokenCount != 0) {
						Double x = 1 / Double
								.valueOf(NLP_Bigram_totalBuisnessCount
										+ NLP_Bigram_vocabSet.size());
						buisnessScore = buisnessScore + Math.log(x)
								/ Math.log(2);
						buisnessVocabTokenCount--;
					}
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
			String[] tokens = line.split(splitter);
			for (int i = 0; i < tokens.length - 1; i++) {
				if (tokens[i].equals("") || tokens[i + 1].equals(""))
					continue;

				if (existingTokens.contains(tokens[i] + " " + tokens[i + 1])) {
					Integer count = docVocab.get(tokens[i] + " "
							+ tokens[i + 1]);
					count++;
					docVocab.put(tokens[i] + " " + tokens[i + 1], count);
				} else {
					docVocab.put(tokens[i] + " " + tokens[i + 1], 1);
				}

			}

		}
		reader.close();
		return docVocab;
	}

	public static void learnFromPolictical(File politicalFolder)
			throws IOException {
		// list all the files in directory
		String files[] = politicalFolder.list();
		for (String file : files) {
			politicalDocCount++;
			File newFile = new File(politicalFolder, file);
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
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
				Set<String> existingTokens = NLP_Bigram_politicalVocab.keySet();
				String[] tokens = line.split(splitter);
				for (int i = 0; i < tokens.length - 1; i++) {
					if (tokens[i].equals("") || tokens[i + 1].equals(""))
						continue;

					NLP_Bigram_totalPoliticalCount++;
					if (existingTokens
							.contains(tokens[i] + " " + tokens[i + 1])) {
						Integer count = NLP_Bigram_politicalVocab.get(tokens[i]
								+ " " + tokens[i + 1]);
						count++;
						NLP_Bigram_politicalVocab.put(tokens[i] + " "
								+ tokens[i + 1], count);
					} else {
						NLP_Bigram_politicalVocab.put(tokens[i] + " "
								+ tokens[i + 1], 1);
					}
				}
			}
			reader.close();
		}

	}

	public static void learnFromSports(File sportsFolder) throws IOException {
		// list all the files in directory
		String files[] = sportsFolder.list();
		for (String file : files) {
			sportsDocCount++;
			File newFile = new File(sportsFolder, file);
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			String line;
			while ((line = reader.readLine()) != null) {

				line = line.replace("-", "");
				line = line.replace(",", "");
				line = line.replace(".", "");
				line = line.replace("/", "");
				line = line.replace(":", "");
				Set<String> existingTokens = NLP_Bigram_sportsVocab.keySet();
				String[] tokens = line.split(splitter);
				for (int i = 0; i < tokens.length - 1; i++) {
					if (tokens[i].equals("") || tokens[i + 1].equals(""))
						continue;

					NLP_Bigram_totalSportsCount++;
					if (existingTokens
							.contains(tokens[i] + " " + tokens[i + 1])) {
						Integer count = NLP_Bigram_sportsVocab.get(tokens[i]
								+ " " + tokens[i + 1]);
						count++;
						NLP_Bigram_sportsVocab.put(tokens[i] + " "
								+ tokens[i + 1], count);
					} else {
						NLP_Bigram_sportsVocab.put(tokens[i] + " "
								+ tokens[i + 1], 1);
					}

				}

			}
			reader.close();
		}
		System.out.println();
	}

	public static void learnFromBuisness(File buisnessFolder)
			throws IOException {
		// list all the files in directory
		String files[] = buisnessFolder.list();
		for (String file : files) {
			buisnessDocCount++;
			File newFile = new File(buisnessFolder, file);
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			String line;
			while ((line = reader.readLine()) != null) {

				line = line.replace("-", "");
				line = line.replace(",", "");
				line = line.replace(".", "");
				line = line.replace("/", "");
				line = line.replace(":", "");
				Set<String> existingTokens = NLP_Bigram_buisnessVocab.keySet();
				String[] tokens = line.split(splitter);
				for (int i = 0; i < tokens.length - 1; i++) {
					if (tokens[i].equals("") || tokens[i + 1].equals(""))
						continue;

					NLP_Bigram_totalBuisnessCount++;
					if (existingTokens
							.contains(tokens[i] + " " + tokens[i + 1])) {
						Integer count = NLP_Bigram_buisnessVocab.get(tokens[i]
								+ " " + tokens[i + 1]);
						count++;
						NLP_Bigram_buisnessVocab.put(tokens[i] + " "
								+ tokens[i + 1], count);
					} else {
						NLP_Bigram_buisnessVocab.put(tokens[i] + " "
								+ tokens[i + 1], 1);
					}

				}

			}
			reader.close();
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
