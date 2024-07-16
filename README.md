# OpenLongVA
An open-source implementation of LongVA for facilitating the large multi-modal model community.

# Construct Datasets for Finetuning
{
        "id": id of the video/image data,
        "video": dir to the video/image data,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n Instruction"
            },
            {
                "from": "gpt",
                "value": "Output"
            },
            {
                "from": "human",
                "value": "<image>\n Instruction"
            },
            {
                "from": "gpt",
                "value": "Output"
            }
        ]
    },
