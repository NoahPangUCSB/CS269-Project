import re
from datasets import load_dataset

def extract_human_prompts(conversation):
    pattern = r'\n\nHuman: (.*?)(?=\n\nAssistant:|$)'
    matches = re.findall(pattern, conversation, re.DOTALL)
    return ' '.join(m.strip() for m in matches)

ds = load_dataset('ethz-spylab/rlhf_trojan_dataset', split='train')

# Check if prompts are identical
mismatches = 0
for i in range(100):
    chosen_prompt = extract_human_prompts(ds[i]['chosen'])
    rejected_prompt = extract_human_prompts(ds[i]['rejected'])

    if chosen_prompt != rejected_prompt:
        mismatches += 1
        print(f'Sample {i}: MISMATCH!')
        print(f'Chosen:   {chosen_prompt[:100]}')
        print(f'Rejected: {rejected_prompt[:100]}')

print(f'\nChecked 100 samples: {mismatches} mismatches')
print(f'Prompts are identical: {mismatches == 0}')
