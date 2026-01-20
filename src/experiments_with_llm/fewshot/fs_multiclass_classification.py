import argparse
import os
import re
import warnings
from typing import Dict, List

import pandas as pd
import rootutils
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

# Setup
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

EXAMPLES = {
    "ElecDeb60to20": [
        "Sentence: we have avoided surrender of principle or territory at the conference table [SEP] In the past seven years, in President Eisenhower's Administration, this situation has been reversed\n"
        "Output: support\n",

        "Sentence: although we are today, as Senator Kennedy has admitted, the strongest nation in the world militarily [SEP] we must increase our strength\n"
        "Output: attack\n",

        "Sentence: You know, people said it was a big risk at that time [SEP] when there was a crisis involving the Mexican peso, again, President Bill Clinton showed bold and dynamic leadership\n"
        "Output: no_relation\n",
    ],

    "MotionPolicyPreference": [
        "Sentence: expresses concern about the impact of the Care Records Service on patient confidentiality\n"
        "Output: Human Rights\n",

        "Sentence: condemns the pursuit of uniformity at the expense of diversity\n"
        "Output: Equality: Positive\n",

        "Sentence: observes that the Health and Safety Act 1974 made it a legal necessity for workplaces to train someone in medical first aid\n"
        "Output: Labour Groups: Positive\n",
    ],

    "ParlVote+": [
        "Sentence: Please stick to the motion.I congratulate the right hon Member for Birkenhead  and thank the Backbench Business Committee for granting us this very timely debate to reconsider the impact on the lowest-paid workers of the proposed changes to tax credits and to call for the Government to bring mitigation proposals to this House. Early next year it is the centenary of the birth of Harold Wilson. That Huddersfield lad coined the phrase, “A week is a long time in politics.” A lot of ermine and a flood of emails have flowed under the bridge since I signed this motion last week. I want to make it clear from the start that I absolutely support the Chancellor in getting Britain to live within its means. In fact, I often suggest to folk back home in Yorkshire who are talking about austerity that we should replace it with the phrase “living within our means”. That brings a whole new meaning to the campaign slogan “anti-living within your means”. Since last week, many constituents have echoed my position. To follow the style of the Leader of the Opposition, Martin from Holme Valley says that he agrees with the shift from tax credits to increased pay but shares my concern about the transitional impact of the changes. Bob from Salendine Nook says that he understands the point I make about employers underpaying staff and agrees with me on the need to reconsider the pace of change. Nicola from Oakes says that she agrees that the tax credits system is imperfect, as is the whole benefits system. She says that she would be better off financially reducing her hours, as she works full time, and that a change to the system needs to be implemented. She says that she feels she is currently being punished by the benefits system for trying to bring home more money by working her way up, and that a single person on income support, disability living allowance, housing benefit and other benefits could, in effect, be paid more in benefits than she brings home, including with her tax credits, to support a family. Dorothy from Marsden says that she fully understands the need for reform. As the motion clearly states, this about the pace and the impact on the lowest-paid workers. I firmly believe that work should always pay. People should always be better off in a job than on benefits. I say that as someone who did not go to university. When I left school, I did a succession of low-paid, part-time jobs before I joined the Royal Air Force at the age of 19, worked my way up, and travelled the world. I am proud that since 2010 unemployment is down by 51% in my constituency. I am proud that youth unemployment is down by more than half. I am proud that there is a net increase of 170 new businesses and there have been over 4,700 new apprenticeship starts. I am proud to say that I have just taken on my first apprentice and that I am paying him the living wage. On Friday 20 November I will hold my latest jobs fair at Holmfirth civic hall, where over 30 local businesses and organisations will be offering quality jobs and apprenticeships. We must build a low-tax, low-welfare, high-wage economy. As a compassionate Conservative, I want to live in a country where everyone has the opportunity of a decent, well-paid job. So let us crack on with it, and let us stand up for working people. I welcome the Chancellor’s announcement that he will lessen the impact on families and will set out these plans in the autumn statement. I hope that he and his Treasury boffins will be listening very carefully to the various suggestions, some of them very inventive, for transitional arrangements. Let us show that Britain can live within its means while, most importantly, looking after the most vulnerable and supporting those who go out and work every day. Several hon Members rose—\n"
        "Output: Welfare State Expansion\n",

        """Sentence: I had begun to doubt myself. Having intended to speak in favour of amendment No. 279, I began to have my doubts when I heard Labour Members question my hon Friend the Member for Westmorland and Lonsdale . However, I have now returned to my former opinion℄even more starkly℄having heard the hon Member for Edinburgh, West , who raises every fear that I had about the Bill. The Bill states: Statutory functions may be conferred on the Scottish Ministers".  Having read the Bill in its totality, I would welcome the addition of the rider: "but not in so far as they might affect England, Wales and Northern Ireland." That seems entirely proper, and everything that the hon Member for Edinburgh, West has said confirms that. It is difficult to confer a general duty on a Member of Parliament to hold the Executive to account when there remains a possibility℄albeit theoretical℄that the Executive are not accountable to him. That is a theoretical argument, but there is a more important argument which has more resonance in terms of public opinion. I would say to the hon and learned Member for Edinburgh, Pentlands  that, in these matters, perceptions are even more important than reality. Even now, I am approached by constituents at meetings and in public℄in a most unwelcome, unsolicited and unhelpful way℄who share their observation that, "Now Scotland is getting its own Parliament, isn't it awful and awkward that we have a Government entirely dominated by Scottish Members?" I have to point out to them that their concern is quite illegitimate and that they have no proper grievance. Those who sit on the Treasury Bench have a mandate and are very properly there. I should hate it if a situation arose in which my constituents had a legitimate grievance, and amendment No. 279 would prevent that from happening. As said, perception is important in these respects, and the Committee has a duty to remove such sources of conflict and public discontent..\n"""
        "Output: Centralisation: Positive\n",

        """Sentence: I speak on behalf of those who are prepared to accept the Government's proposals—or to be bought off, as the hon Member for Kingston and Surbiton  suggests. I shall explain why. I am pleased that we are having this debate. My direct involvement in this Bill began when I chaired the Joint Scrutiny Committee—with the Home Affairs Committee and the Work and Pensions Committee—on it. Perhaps I should not say so as the Chairman of that Committee, but its work showed the value of draft scrutiny, because many other issues were sorted out before the Bill was introduced. I wish that we did that more often. The issue before the House is one of the few outstanding issues on which the Committee took a different view from the Government on what should be in the Bill. I argued the same case on Report and I have seen amendments passed in the other place, and I welcome the efforts that the Minister has made to get us to where we are tonight. I know—and from comments that have been made, the House knows—how much work it has required from him to get us to this position. For some people, who have followed the Bill from the perspective of losing friends or relatives in accidents such as the Marchioness disaster or train crashes, this subject appears to be a late entry into the Bill, which has given rise to concerns that the Bill itself may be threatened by our desire to extend it to cover custody. The Minister has done his best to get us to a position where the principle of covering deaths in custody is covered by the Bill without putting it under threat. He deserves the thanks of the House for that. Why am I prepared to accept this when, from the amendments that I tabled last time, it is evident that my preference would be a straightforward amendment to the Bill? The answer is that I believe that the amendment the Minister has tabled today will trigger a process that will lead inexorably to deaths in custody being brought within the scope of the Bill. Whether or not he feels he has the freedom tonight to talk about time scales, the fact is that the process, once started, will be unstoppable. The Minister is also responsible for prisons, and I do not believe that when he next meets the director of the Prison Service, he will say to him, "I've got you out of that problem for the next few years, so I wouldn't worry about it if I were you. Instead, the Minister will say that deaths in custody could be included at any time and the Prison Service will have to be ready for that. Many of us would accept that, even if the provision had been on the face of the Bill, some delay in commencement would have been necessary to get the Prison Service to face up to its responsibilities in a way that it has not done so far. That is how the conversation will go. I do not wish to prejudge inquests in the pipeline, but there are some whose verdicts could make it untenable for the Government not to bring forward the resolution necessary under the amendment, however much the Government may prefer to consider the statutory ombudsman first and the forum next. I fear that there will be other inquests beyond those. That is the historical record, and the result will be that it will not be tenable for this or any Government not to enact this change. I believe that those of us who, a year or so ago, set out to bring this matter within the scope of the Bill are going to achieve our objective tonight. It is for that reason that I recommend to the House that we support the Government's proposal, which takes us a long way forward. As ever when Governments are asked to take a very different position from the one that they started with, there is a certain choreography about how such matters must be handled, but the amendment in lieu in the end gets us to where we want to go and delivers the result that we want. I am therefore very grateful to the Minister for everything that he has done.\n"""
        "Output: Law and Order: Positive\n",
    ],
    "ArgUNSC": [
        "Sentence: the facilitation of contacts between the warring parties, are a positive sign but they are not enough. [SEP] we will remain vigilant to ensure that support for the separatists finally ceases, that the weakening of the State and the rule of law is checked, and that the ceasefire is fully respected.\n"
        "Output: no_relation\n",

        "Sentence: the separatists trained, supplied and supported by  Russia are launching a full-scale attack on the strategic city of Debaltseve, inside Ukrainian-controlled territory, in blatant violation of the 19 September Minsk ceasefire lines, in an attemp to gain control of a significant rail juncture. [SEP] It is dangerous because Russia continues to train and equip separatists with heavy weapons and to fight by their side, in flagrant violation of the September Minsk agreement, Ukrainian sovereignty and international law.\n"
        "Output: support\n",

        "Sentence: I would repeat that Russia would never refuse to implement any useful document agreed upon during the crisis, including the 21 February agreement. [SEP] Regarding what one colleague said, that Russia is refusing to implement the 17 April Geneva statement\n"
        "Output: attack\n"
    ]
}

INSTRUCTION_PROMPT = {
    "ElecDeb60to20": (
        "You are a relation classification assistant. Classify the sentences separated by [SEP] using the labels: support, attack, no_relation\n\n"
        "{examples}"
        "Sentence: {sentence}\n"
        "Output:"
    ),
    "ArgUNSC": (
        "You are a relation classification assistant. Classify the sentences separated by [SEP] using the labels: support, attack, no_relation\n\n"
        "{examples}"
        "Sentence: {sentence}\n"
        "Output:"
    ),

    "MotionPolicyPreference": (
        "You are a motion policy preference classification assistant. Classify the motions using the following labels:"
        "Military: Positive, Military: Negative, Peace, Internationalism: Positive, "
        "European Union: Positive, European Union: Negative, Human Rights, "
        "Direct Democracy: Positive, Constitutionalism: Positive, Constitutionalism: Negative, "
        "Decentralisation: Positive, Centralisation: Positive, Governative and Administrative Efficiency, "
        "Political Corruption, Political Authority: Party, Free Market Economy, Incentives: Positive, "
        "Market Regulation, Economic Planning, Corporatism/Mixed Economy, Protectionism: Negative, "
        "Economic Goals, Keynesian Demand Management, Economic Growth: Positive, "
        "Technology and Infrastructure: Positive, Controlled Economy, Nationalisation, "
        "Economic Orthodoxy, Environmental Protection, Culture: Positive, Equality: Positive, "
        "Welfare State Expansion, Welfare State Limitation, Education Expansion, "
        "National Way of Life: Positive, National Way of Life: Negative, "
        "Law and Order: Positive, Civic Mindedness: Positive, Multiculturalism: Positive, "
        "Multiculturalism: Negative, Labour Groups: Positive, Agriculture and Farmers: Positive, "
        "Middle Class and Professional Groups, Underprivileged Minority Groups, "
        "Non-economic Demographic Groups\n\n"
        "{examples}"
        "Sentence: {sentence}\n"
        "Output:"
    ),

    "ParlVote+": (
        "You are a motion policy preference classification assistant. Classify the motions using the following labels:"
        "Military: Positive, Military: Negative, Peace, European Union: Positive, "
        "European Union: Negative, Human Rights, Direct Democracy: Positive, "
        "Constitutionalism: Positive, Constitutionalism: Negative, Decentralisation: Positive, "
        "Centralisation: Positive, Political Corruption, Political Authority: Party, "
        "Political Authority: Personal, Free Market Economy, Incentives: Positive, "
        "Market Regulation, Technology: Positive, Nationalisation, Environmental Protection, "
        "Equality: Positive, Welfare State Expansion, Welfare State Limitation, "
        "Education Expansion, Education Limitation, Immigration: Negative, Immigration: Positive, "
        "Traditional Morality: Positive, Traditional Morality: Negative, "
        "Law and Order: Positive, Law and Order: Negative, "
        "Labour Groups: Positive, Labour Groups: Negative, "
        "Underprivileged Minority Groups\n\n"
        "{examples}"
        "Sentence: {sentence}\n"
        "Output:"
    ),
}


def load_model(model_name: str, quantize_4bit: bool = True):
    """Load model + tokenizer with optional 4-bit quantization."""
    if quantize_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if quantize_4bit else None,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        do_sample=False,
        temperature=0.0,
    )

    return pipe


def build_prompt(dataset: str, sentence: str) -> str:
    examples = "".join(EXAMPLES[dataset])
    template = INSTRUCTION_PROMPT[dataset]
    return template.format(examples=examples, sentence=sentence)


def extract_label_relations(text: str) -> str:
    match = re.search(r"(support|attack|no_relation|no relation|no rel)", text.lower())
    return match.group(1) if match else "none"


def extract_label_MotionPolicyPreference(text: str) -> str:
    match = re.search(
        r"(Agriculture and Farmers: Positive|Centralisation: Positive|Civic Mindedness: Positive|Constitutionalism: Negative|Constitutionalism: Positive|Controlled Economy|Corporatism/Mixed Economy|Culture: Positive|Decentralisation: Positive|Direct Democracy: Positive|Economic Goals|Economic Growth: Positive|Economic Orthodoxy|Economic Planning|Education Expansion|Environmental Protection|Equality: Positive|European Union: Negative|European Union: Positive|Free Market Economy|Governative and Administrative Efficiency|Human Rights|Incentives: Positive|Internationalism: Positive|Keynesian Demand Management|Labour Groups: Positive|Law and Order: Positive|Market Regulation|Middle Class and Professional Groups|Military: Negative|Military: Positive|Multiculturalism: Negative|Multiculturalism: Positive|National Way of Life: Negative|National Way of Life: Positive|Nationalisation|Non-economic Demographic Groups|Peace|Political Authority: Party|Political Corruption|Protectionism: Negative|Technology and Infrastructure: Positive|Underprivileged Minority Groups|Welfare State Expansion|Welfare State Limitation)",
        text.lower())
    return match.group(1) if match else "none"


def extract_label_ParlVotePlus(text: str) -> str:
    match = re.search(
        r"(Centralisation: Positive|Constitutionalism: Negative|Constitutionalism: Positive|Decentralisation: Positive|Direct Democracy: Positive|Education Expansion|Education Limitation|Environmental Protection|Equality: Positive|European Union: Negative|European Union: Positive|Free Market Economy|Human Rights|Immigration: Negative|Immigration: Positive|Incentives: Positive|Labour Groups: Negative|Labour Groups: Positive|Law and Order: Negative|Law and Order: Positive|Market Regulation|Military: Negative|Military: Positive|Nationalisation|Peace|Political Authority: Party|Political Authority: Personal|Political Corruption|Technology: Positive|Traditional Morality: Negative|Traditional Morality: Positive|Underprivileged Minority Groups|Welfare State Expansion|Welfare State Limitation)",
        text.lower())
    return match.group(1) if match else "none"


def compute_metrics(gold: List[str], pred: List[str]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(gold, pred),
        "precision": precision_score(gold, pred, average="macro", zero_division=0),
        "recall": recall_score(gold, pred, average="macro", zero_division=0),
        "f1": f1_score(gold, pred, average="macro", zero_division=0),
    }


def run(args):
    df = pd.read_csv(f"data/multi_class_classification/{args.dataset}/test.csv")
    pipe = load_model(args.model)

    predictions = []
    predictions_text = []
    gold_labels = df[args.label_col].tolist()
    sentences = df[args.text_col].tolist()

    extract_label = {
        'ElecDeb60to20': extract_label_relations,
        'MotionPolicyPreference': extract_label_MotionPolicyPreference,
        'ParlVote+': extract_label_ParlVotePlus,
        'ArgUNSC': extract_label_relations
    }

    for s in tqdm(sentences, desc="Classifying"):
        prompt = build_prompt(args.dataset, s)
        output = pipe(prompt)[0]["generated_text"]
        label = extract_label.get(args.dataset)(output)
        predictions.append(label)
        predictions_text.append(output)

    df["prediction"] = predictions
    df['predictions_text'] = predictions_text
    os.makedirs(f"logs/{args.model}/{args.dataset}", exist_ok=True)
    out_file = f"logs/{args.model}/{args.dataset}/few_shot_multi_class_classification.csv"

    df.to_csv(out_file, index=False)
    print(f"\nSaved predictions to: {out_file}")

    # metrics = compute_metrics(gold_labels, predictions)
    # print("###################### RESULTS ######################")
    # print(f"\nEvaluation - multiclass classification - {args.dataset}:")
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--dataset", type=str, choices=EXAMPLES.keys(), default="ParlVote+")

    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label_value")

    args = parser.parse_args()
    run(args)
