import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v2-finetuned-korquad")
model = AutoModelForQuestionAnswering.from_pretrained("monologg/koelectra-base-v2-finetuned-korquad")


def answer_question(question, contents):
    """
    질문, 본문을 파라미터로 받아 본문에서 질문에대한 답변을 뽑아낸다

    :param question:
    :param contents:
    :return: answer
    """

    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, contents)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                                     token_type_ids=torch.tensor(
                                         [segment_ids]))  # The segment IDs to differentiate question from contents

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')


if __name__ == '__main__':
    contents = """
    한국어 위키백과(영어: Korean Wikipedia)는 한국어로 운영되는 위키백과의 다언어판 가운데 하나로서, 2002년 10월 11일에 시작되었다. 
    또한 현재 한국어 위키백과에는 넘겨주기, 토론, 그림 등 페이지로 불리는 모든 문서를 포함하면 총 2,652,208개가 수록되어 있으며, 넘겨주기를 포함한 일반 문서 수는 1,292,587개,[1] 
    그중 넘겨주기, 막다른 문서를 제외한 일반 문서 수는 580,132개이다. 위키백과 언어판 중에서 23번째로 크며[2] 한국어로 된 위키 중에서는 1위 규모[a]이며, 기호는 ko이다.
    """

    question = "총 문서 수는 몇개인가?"

    answer_question(question, contents)