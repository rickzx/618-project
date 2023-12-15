import argparse

from mlc_chat import ChatConfig, ChatModule

GATSBY_CHAP1_FIRST_THREE_PARAGRAPHS = """
In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since.

"Whenever you feel like criticizing any one," he told me, "just remember that all the people in this world haven't had the advantages that you've had."

He didn't say any more but we've always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that. In consequence I'm inclined to reserve all judgments, a habit that has opened up many curious natures to me and also made me the victim of not a few veteran bores. The abnormal mind is quick to detect and attach itself to this quality when it appears in a normal person, and so it came about that in college I was unjustly accused of being a politician, because I was privy to the secret griefs of wild, unknown men. Most of the confidences were unsought—frequently I have feigned sleep, preoccupation, or a hostile levity when I realized by some unmistakable sign that an intimate revelation was quivering on the horizon—for the intimate revelations of young men or at least the terms in which they express them are usually plagiaristic and marred by obvious suppressions. Reserving judgments is a matter of infinite hope. I am still a little afraid of missing something if I forget that, as my father snobbishly suggested, and I snobbishly repeat, a sense of the fundamental decencies is parcelled out unequally at birth.
"""

FULL_GETTYSBURG_ADDRESS = """
Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.

But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth.
"""

parser = argparse.ArgumentParser(description="Benchmark an MLC LLM ChatModule.")
parser.add_argument(
    "--model",
    type=str,
    help="""The model folder after compiling with MLC-LLM build process. The parameter can either
    be the model name with its quantization scheme (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a
    full path to the model folder. In the former case, we will use the provided name to search for
    the model folder over possible paths.""",
    default="./dist/gpt2-xl-q0f32/params/",
    required=False,
)
parser.add_argument(
    "--model-lib",
    type=str,
    help="""The compiled model library. In MLC LLM, an LLM is compiled to a shared or static
    library (.so or .a), which contains GPU computation to efficiently run the LLM. MLC Chat,
    as the runtime of MLC LLM, depends on the compiled model library to generate tokens.
    """,
    required=True,
)
parser.add_argument(
    "--device",
    type=str,
    help="""The description of the device to run on. User should provide a string in the form of
    'device_name:device_id' or 'device_name', where 'device_name' is one of 'cuda', 'metal',
    'vulkan', 'rocm', 'opencl', and 'device_id' is the device id to run on. If no 'device_id' is
    provided, it will be set to 0 by default.
    """,
    required=True,
)
parser.add_argument(
    "--prompt",
    type=str,
    help="The prompt to generate from.",
    default=FULL_GETTYSBURG_ADDRESS,
    # default=GATSBY_CHAP1_FIRST_THREE_PARAGRAPHS,
    required=False,
)
parser.add_argument(
    "--generate-length",
    type=int,
    help="The length (numer of tokens) of the generated text.",
    default=512,
    required=False,
)


def main():
    """The main function that runs the benchmarking."""

    args = parser.parse_args()

    chat_module = ChatModule(
        model=args.model,
        device=args.device,
        model_lib_path=args.model_lib,
    )
    output = chat_module.benchmark_generate(args.prompt, generate_length=args.generate_length)
    print(f"Generated text:\n{output}\n")
    print(f"Statistics: {chat_module.stats(verbose=True)}")


if __name__ == "__main__":
    main()
