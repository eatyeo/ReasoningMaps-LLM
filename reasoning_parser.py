import re
import networkx as nx
import matplotlib.pyplot as plt

class ReasoningMap:
    """
    This class takes the raw text output from the LLM and the original
    LSAT problem, parses the text, and builds a NetworkX graph
    representing the logical steps.
    """
    def __init__(self, raw_llm_text, lsat_problem):
        self.raw_text = raw_llm_text
        self.problem = lsat_problem
        
        self.graph = nx.DiGraph() # DiGraph = Directed Graph
        self.steps = {} # Store the parsed text for each step
        self.is_correct = False # Was the LLM's final answer correct?
        self.llm_answer = None # Will be 0 for A, 1 for B, etc.

    def parse_reasoning(self):
        """
        Uses Regular Expressions (regex) to find the sections
        of the LLM's output (e.g., "1. Argument Breakdown", "4. Final Conclusion")
        and stores them.
        """

        pattern = re.compile(
            r"(\d+)\.?\s*\**([^:]+):\**\s*(.*?)(?=\n\d+\.?\s*\**|\Z)", 
            re.DOTALL
        )
        
        matches = pattern.finditer(self.raw_text)
        
        for match in matches:
            step_title = match.group(2).strip()
            step_content = match.group(3).strip()
            
            # Standardize titles (e.g., "Argument Breakdown")
            if "argument breakdown" in step_title.lower():
                self.steps["Argument Breakdown"] = step_content
            elif "question analysis" in step_title.lower():
                self.steps["Question Analysis"] = step_content
            elif "strategic evaluation" in step_title.lower():
                self.steps["Strategic Evaluation"] = step_content
            elif "final conclusion" in step_title.lower():
                self.steps["Final Conclusion"] = step_content
            else:
                # Store other steps too
                self.steps[step_title] = step_content

        if not self.steps:
            print("--- PARSING FAILED ---")
            print("Could not find any steps in the LLM output.")
            print("Raw text was:")
            print(self.raw_text)
            print("----------------------")


    def analyze_correctness(self):
        """
        Checks the LLM's "Final Conclusion" against the problem's
        ground-truth 'label' to see if the LLM was correct.
        """
        conclusion_text = self.steps.get("Final Conclusion", "")
        
        if not conclusion_text:
            print("Warning: Could not find 'Final Conclusion' step.")
            return

        # Look for the answer letter (A, B, C, D, or E)
        # This regex looks for (A), (A., A:, A. or just A
        match = re.search(r"\b\(?([A-E])\)?[:\.]?\b", conclusion_text)
        
        if match:
            letter = match.group(1)
            # Convert letter (A=0, B=1, etc.)
            self.llm_answer = ord(letter) - ord('A')
            
            if self.llm_answer == self.problem['label']:
                self.is_correct = True
        else:
            print(f"Warning: Could not find answer letter (A-E) in conclusion: '{conclusion_text}'")
        
    def build_graph(self):
        """
        Builds the actual NetworkX graph from the parsed steps.
        This is the "reasoning map."
        """
        if not self.steps:
            print("Cannot build graph, no steps were parsed.")
            return

        # 1. Add node for the *Problem Context*
        self.graph.add_node("Context", label="Problem Context")
        
        # 2. Add nodes for each step
        # Use standardized keys
        std_steps = ["Argument Breakdown", "Question Analysis", "Strategic Evaluation", "Final Conclusion"]
        for step in std_steps:
            if step in self.steps:
                self.graph.add_node(step, label=step)
            
        # 3. Create the logical flow (edges)
        if "Argument Breakdown" in self.graph:
            self.graph.add_edge("Context", "Argument Breakdown")
        if "Argument Breakdown" in self.graph and "Question Analysis" in self.graph:
            self.graph.add_edge("Argument Breakdown", "Question Analysis")
        if "Question Analysis" in self.graph and "Strategic Evaluation" in self.graph:
            self.graph.add_edge("Question Analysis", "Strategic Evaluation")
        if "Strategic Evaluation" in self.graph and "Final Conclusion" in self.graph:
            self.graph.add_edge("Strategic Evaluation", "Final Conclusion")

    def visualize(self, save_path="reasoning_map.png"):
        """
        Draws the graph using Matplotlib and saves it to a file.
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, cannot visualize.")
            return

        plt.figure(figsize=(12, 8))
        
        # Determine node colors
        color_map = []
        for node in self.graph:
            if node == "Final Conclusion":
                # Check llm_answer is not None before comparing
                if self.is_correct:
                    color_map.append("#aaffaa") # Light Green
                else:
                    color_map.append("#ffaaaa") # Light Red
            elif node == "Context":
                color_map.append("#cceeff") # Light blue
            else:
                color_map.append("#ffddc1") # Light orange
        
        # ---------------------
        # Manually define the positions for a clean, top-down flow
        pos = {}
        if "Context" in self.graph:
            pos["Context"] = (0.5, 0.9)
        if "Argument Breakdown" in self.graph:
            pos["Argument Breakdown"] = (0.5, 0.7)
        if "Question Analysis" in self.graph:
            pos["Question Analysis"] = (0.5, 0.5)
        if "Strategic Evaluation" in self.graph:
            pos["Strategic Evaluation"] = (0.5, 0.3)
        if "Final Conclusion" in self.graph:
            pos["Final Conclusion"] = (0.5, 0.1)

        # Fallback if for some reason a node was missed
        for node in self.graph.nodes():
            if node not in pos:
                print(f"Warning: Node '{node}' missing from manual layout. Adding.")
                pos[node] = (0.1, 0.1)
        # ---------------------

        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color=color_map,
            node_size=3000,
            font_size=10,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
            font_weight='bold'
        )
        
        plt.title(f"Reasoning Map for: {self.problem['id_string']}", size=15)
        
        # Save the file
        plt.savefig(save_path)
        print(f"Map saved to {save_path}")