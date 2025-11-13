import re
import networkx as nx
import matplotlib.pyplot as plt

class ReasoningMap:
    def __init__(self, raw_llm_text, lsat_problem):
        self.raw_text = raw_llm_text
        self.problem = lsat_problem
        self.graph = nx.DiGraph()
        self.steps = {} # Store the parsed text for each step
        self.is_correct = False
        self.llm_answer = None
        self.step_titles = [
            "Argument Breakdown", 
            "Question Analysis", 
            "Strategic Evaluation", 
            "Final Conclusion"
        ]

    def parse_reasoning(self):
        """
        A more robust parser. Instead of one complex regex, this looks
        for the step titles (e.g., "Argument Breakdown") and captures
        all text until the next title or the end of the string.
        """
        text = self.raw_text
        
        for i, title in enumerate(self.step_titles):
            # Create a regex to find the title.
            # re.IGNORECASE makes it case-insensitive.
            # It looks for the title, followed by an optional ':', ' ', or '**'
            start_pattern = re.compile(rf"{re.escape(title)}:?\s*\n?", re.IGNORECASE)
            match = start_pattern.search(text)
            
            if not match:
                continue # Could not find this step

            # This is the text *after* the title
            content_start = match.end()
            content = ""

            # Now, find where this content *ends*
            # It ends at the start of the *next* step, or the end of the string
            end_match = None
            if i + 1 < len(self.step_titles):
                next_title = self.step_titles[i+1]
                next_pattern = re.compile(rf"{re.escape(next_title)}:?\s*\n?", re.IGNORECASE)
                end_match = next_pattern.search(text, content_start) # Search *after* our match

            if end_match:
                content = text[content_start : end_match.start()]
            else:
                # It's the last step, so just take all remaining text
                content = text[content_start:]
            
            self.steps[title] = content.strip()

    def analyze_correctness(self):
        """
        Checks the LLM's "Final Conclusion" against the problem's
        ground-truth 'label' to see if the LLM was correct.
        """
        conclusion_text = self.steps.get("Final Conclusion")
        search_text = ""
        
        if conclusion_text:
            # If step is found, search inside it
            search_text = conclusion_text
        else:
            # If the parser couldn't find the "Final Conclusion" step,
            # fall back to searching the *entire raw text*.
            # This handles cases where the LLM forgets the header.
            print(f"Warning: Parser could not find 'Final Conclusion' step for {self.problem['id_string']}. Searching full text.")
            search_text = self.raw_text

        # Look for the answer letter (A, B, C, D, or E)
        # Tries to find "The answer is A" or "Conclusion: (B)"
        match = re.search(r"(?:answer is|conclusion:)\s*\(?([A-E])\)?", search_text, re.IGNORECASE)
        
        if not match:
             # Fallback: just find the last single letter in the text
             matches = re.findall(r"\b([A-E])\b", search_text)
             if matches:
                letter = matches[-1].upper() # Get the last one
                self.llm_answer = ord(letter) - ord('A')
             else:
                print(f"Warning: Could not parse answer letter from text for {self.problem['id_string']}.")
                return # Give up
        else:
            letter = match.group(1).upper()
            self.llm_answer = ord(letter) - ord('A')
        
        # Check correctness of answer
        if self.llm_answer is not None and self.llm_answer == self.problem['label']:
            self.is_correct = True
        
    def build_graph(self):
        """
        Builds the actual NetworkX graph from the parsed steps.
        """
        self.graph.add_node("Context", label="Problem Context")
        
        last_node = "Context"
        for title in self.step_titles:
            if title in self.steps:
                self.graph.add_node(title, label=title)
                self.graph.add_edge(last_node, title)
                last_node = title
            else:
                # Add a "Missing" node to show the break in the chain
                missing_title = f"Missing:\n{title}"
                self.graph.add_node(missing_title, label=missing_title)
                self.graph.add_edge(last_node, missing_title)
                last_node = missing_title


    def visualize(self, save_path="reasoning_map.png"):
        """
        Draws the graph using Matplotlib and saves it to a file.
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, cannot visualize.")
            return

        plt.figure(figsize=(10, 10))
        
        # Create a fixed, top-to-bottom layout
        pos = {}
        node_list = list(self.graph.nodes())
        y = len(node_list)
        for node in node_list:
            pos[node] = (0, y)
            y -= 1
            
        # Determine node colors
        color_map = []
        for node in self.graph:
            if "Final Conclusion" in node and node.startswith("Missing"):
                color_map.append("#ffcccc") # Light red (missing)
            elif "Final Conclusion" in node:
                color_map.append("#ccffcc" if self.is_correct else "#ffcccc") # Green/Red
            elif "Context" in node:
                color_map.append("#cceeff") # Light blue
            elif "Missing" in node:
                color_map.append("#f0f0f0") # Grey
            else:
                color_map.append("#ffddc1") # Light orange

        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color=color_map,
            node_size=6000,
            font_size=10,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
            node_shape='o',
            font_weight='bold',
            labels=nx.get_node_attributes(self.graph, 'label')
        )
        
        plt.title(f"Reasoning Map for: {self.problem['id_string']}", size=15)
        
        # Save the file
        plt.savefig(save_path)
        plt.close() # Close the plot to save memory
        print(f"Map saved to {save_path}")