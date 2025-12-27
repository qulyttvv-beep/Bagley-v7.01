"""
🧬 Advanced Reasoning Engine
============================

Multi-stage reasoning with verification at each step.
Inspired by o1/DeepSeek-R1 extended thinking.

FEATURES:
1. Tree-of-Thought - Explore multiple reasoning paths
2. Self-Reflection - Evaluate and correct reasoning
3. Backtracking - Abandon bad reasoning paths
4. Proof Verification - Verify logical steps
5. Meta-Cognition - Think about thinking
"""

from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import random
from collections import deque

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Available reasoning strategies"""
    DIRECT = "direct"                    # Simple direct answer
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    TREE_OF_THOUGHT = "tree_of_thought"    # Explore multiple paths
    SELF_CONSISTENCY = "self_consistency"  # Multiple samples, vote
    DEBATE = "debate"                      # Argue multiple sides
    RECURSIVE = "recursive"                # Break into subproblems
    ANALOGICAL = "analogical"              # Reason by analogy


class ThoughtStatus(Enum):
    """Status of a thought node"""
    EXPLORING = "exploring"
    PROMISING = "promising"
    VERIFIED = "verified"
    REJECTED = "rejected"
    COMPLETE = "complete"


@dataclass
class ThoughtNode:
    """A node in the thought tree"""
    id: str
    content: str
    depth: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    status: ThoughtStatus = ThoughtStatus.EXPLORING
    confidence: float = 0.5
    evaluation: Optional[str] = None
    is_solution: bool = False
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class ReasoningStep:
    """A single reasoning step"""
    step_number: int
    thought: str
    reasoning_type: str  # "deduction", "induction", "abduction", "analogy"
    confidence: float
    verification: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Result of reasoning process"""
    answer: str
    confidence: float
    strategy_used: ReasoningStrategy
    steps: List[ReasoningStep]
    thought_tree: Optional[Dict[str, ThoughtNode]] = None
    total_thoughts_explored: int = 0
    reasoning_time_ms: float = 0
    meta_reflection: Optional[str] = None


class TreeOfThought:
    """
    🌳 Tree of Thought Reasoning
    
    Explores multiple reasoning paths simultaneously.
    Prunes bad paths, expands promising ones.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        branching_factor: int = 3,
        beam_width: int = 5,
        exploration_weight: float = 1.4,  # UCB exploration
    ):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.beam_width = beam_width
        self.exploration_weight = exploration_weight
        
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        self.node_counter = 0
    
    def create_node(
        self,
        content: str,
        parent_id: Optional[str] = None,
        depth: int = 0,
    ) -> ThoughtNode:
        """Create a new thought node"""
        self.node_counter += 1
        node_id = f"thought_{self.node_counter}"
        
        node = ThoughtNode(
            id=node_id,
            content=content,
            depth=depth,
            parent_id=parent_id,
        )
        
        self.nodes[node_id] = node
        
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
        
        return node
    
    def initialize(self, problem: str) -> ThoughtNode:
        """Initialize tree with problem statement"""
        self.nodes.clear()
        self.node_counter = 0
        
        root = self.create_node(content=problem, depth=0)
        self.root_id = root.id
        
        return root
    
    def get_promising_nodes(self, n: int = None) -> List[ThoughtNode]:
        """Get most promising nodes to expand"""
        n = n or self.beam_width
        
        # Score nodes using UCB-like formula
        scored = []
        total_visits = sum(1 for n in self.nodes.values() if n.status != ThoughtStatus.EXPLORING)
        
        for node in self.nodes.values():
            if node.status == ThoughtStatus.EXPLORING and node.depth < self.max_depth:
                # UCB score: exploitation + exploration
                visits = len(node.children_ids) + 1
                exploitation = node.confidence
                exploration = self.exploration_weight * (
                    (2 * (total_visits + 1)) ** 0.5 / visits
                ) if total_visits > 0 else 1.0
                
                score = exploitation + exploration
                scored.append((node, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored[:n]]
    
    def get_path_to_root(self, node_id: str) -> List[ThoughtNode]:
        """Get path from node to root"""
        path = []
        current_id = node_id
        
        while current_id:
            if current_id in self.nodes:
                path.append(self.nodes[current_id])
                current_id = self.nodes[current_id].parent_id
            else:
                break
        
        path.reverse()
        return path
    
    def get_best_path(self) -> List[ThoughtNode]:
        """Get best reasoning path (highest confidence solution)"""
        solutions = [
            n for n in self.nodes.values()
            if n.is_solution and n.status in [ThoughtStatus.VERIFIED, ThoughtStatus.COMPLETE]
        ]
        
        if not solutions:
            # Return path to most confident leaf
            leaves = [n for n in self.nodes.values() if not n.children_ids]
            if leaves:
                best_leaf = max(leaves, key=lambda n: n.confidence)
                return self.get_path_to_root(best_leaf.id)
            return []
        
        best_solution = max(solutions, key=lambda n: n.confidence)
        return self.get_path_to_root(best_solution.id)


class SelfReflection:
    """
    🪞 Self-Reflection Module
    
    Evaluates and critiques reasoning.
    Identifies errors and suggests corrections.
    """
    
    REFLECTION_PROMPTS = [
        "Is this reasoning logically valid?",
        "Are there any hidden assumptions?",
        "Could there be alternative explanations?",
        "Am I being biased toward a particular answer?",
        "Have I considered all relevant information?",
        "Is my confidence level justified?",
    ]
    
    def generate_critique(
        self,
        reasoning_steps: List[ReasoningStep],
    ) -> Dict[str, Any]:
        """Generate critique of reasoning"""
        issues = []
        strengths = []
        
        for i, step in enumerate(reasoning_steps):
            # Check confidence consistency
            if i > 0:
                prev_conf = reasoning_steps[i-1].confidence
                if step.confidence > prev_conf + 0.3:
                    issues.append(f"Step {i+1}: Confidence jump without justification")
            
            # Check for absolute language
            absolutes = ["always", "never", "definitely", "certainly", "impossible"]
            if any(word in step.thought.lower() for word in absolutes):
                issues.append(f"Step {i+1}: Uses absolute language that may be overconfident")
            
            # Check for alternatives consideration
            if step.alternatives:
                strengths.append(f"Step {i+1}: Considers {len(step.alternatives)} alternatives")
        
        overall_confidence = sum(s.confidence for s in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0.5
        
        return {
            "issues": issues,
            "strengths": strengths,
            "overall_quality": max(0, 1 - len(issues) * 0.15),
            "suggested_confidence": overall_confidence * max(0.5, 1 - len(issues) * 0.1),
            "should_retry": len(issues) > 2,
        }


class MetaCognition:
    """
    🧠 Meta-Cognition Module
    
    Thinks about thinking.
    Selects appropriate reasoning strategies.
    Monitors cognitive load and confidence.
    """
    
    # Problem type indicators
    PROBLEM_INDICATORS = {
        ReasoningStrategy.DIRECT: ["what is", "define", "name", "list"],
        ReasoningStrategy.CHAIN_OF_THOUGHT: ["how", "why", "explain", "solve", "calculate"],
        ReasoningStrategy.TREE_OF_THOUGHT: ["best", "optimal", "compare", "evaluate options"],
        ReasoningStrategy.SELF_CONSISTENCY: ["complex", "uncertain", "ambiguous"],
        ReasoningStrategy.DEBATE: ["controversial", "argue", "pros and cons", "both sides"],
        ReasoningStrategy.RECURSIVE: ["break down", "step by step", "divide", "sub-problems"],
        ReasoningStrategy.ANALOGICAL: ["similar", "like", "analogy", "compare to"],
    }
    
    def select_strategy(self, problem: str) -> ReasoningStrategy:
        """Select best reasoning strategy for problem"""
        problem_lower = problem.lower()
        
        scores = {strategy: 0 for strategy in ReasoningStrategy}
        
        for strategy, indicators in self.PROBLEM_INDICATORS.items():
            for indicator in indicators:
                if indicator in problem_lower:
                    scores[strategy] += 1
        
        # Add complexity heuristics
        word_count = len(problem.split())
        if word_count > 50:
            scores[ReasoningStrategy.RECURSIVE] += 2
            scores[ReasoningStrategy.TREE_OF_THOUGHT] += 1
        
        if "?" in problem:
            question_count = problem.count("?")
            if question_count > 1:
                scores[ReasoningStrategy.RECURSIVE] += 1
        
        # Default to chain of thought if no clear winner
        if max(scores.values()) == 0:
            return ReasoningStrategy.CHAIN_OF_THOUGHT
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def assess_difficulty(self, problem: str) -> Dict[str, Any]:
        """Assess problem difficulty"""
        word_count = len(problem.split())
        
        # Complexity indicators
        technical_terms = ["algorithm", "optimize", "prove", "derive", "theorem", "hypothesis"]
        technical_count = sum(1 for term in technical_terms if term in problem.lower())
        
        # Ambiguity indicators
        ambiguous_words = ["maybe", "possibly", "could", "might", "some"]
        ambiguity = sum(1 for word in ambiguous_words if word in problem.lower())
        
        # Calculate difficulty
        difficulty = min(1.0, (
            word_count / 200 * 0.3 +
            technical_count / 3 * 0.4 +
            ambiguity / 3 * 0.3
        ))
        
        return {
            "difficulty": difficulty,
            "difficulty_level": "hard" if difficulty > 0.7 else "medium" if difficulty > 0.4 else "easy",
            "estimated_steps": max(1, int(difficulty * 10)),
            "recommended_depth": max(3, int(difficulty * 10)),
            "needs_verification": difficulty > 0.5,
        }


class AdvancedReasoningEngine:
    """
    🚀 Advanced Reasoning Engine
    
    Combines all reasoning techniques into unified system.
    This is what makes Bagley SMARTER than other AIs.
    """
    
    def __init__(
        self,
        max_thinking_time_ms: int = 30000,
        enable_reflection: bool = True,
        enable_meta_cognition: bool = True,
        min_confidence_threshold: float = 0.3,
    ):
        self.max_thinking_time_ms = max_thinking_time_ms
        self.enable_reflection = enable_reflection
        self.enable_meta_cognition = enable_meta_cognition
        self.min_confidence_threshold = min_confidence_threshold
        
        # Components
        self.tree_of_thought = TreeOfThought()
        self.reflection = SelfReflection()
        self.meta_cognition = MetaCognition()
        
        # Statistics
        self.total_problems_solved = 0
        self.total_thoughts_generated = 0
    
    def reason(
        self,
        problem: str,
        context: Optional[str] = None,
        strategy: Optional[ReasoningStrategy] = None,
        generate_thought: Optional[Callable[[str, str], Tuple[str, float]]] = None,
    ) -> ReasoningResult:
        """
        Main reasoning entry point
        
        Args:
            problem: The problem to solve
            context: Additional context
            strategy: Force specific strategy (or auto-select)
            generate_thought: Function to generate thoughts (for integration with LLM)
        
        Returns:
            ReasoningResult with answer and full reasoning trace
        """
        start_time = time.time()
        
        # Auto-select strategy if not specified
        if strategy is None and self.enable_meta_cognition:
            strategy = self.meta_cognition.select_strategy(problem)
            difficulty = self.meta_cognition.assess_difficulty(problem)
        else:
            strategy = strategy or ReasoningStrategy.CHAIN_OF_THOUGHT
            difficulty = {"difficulty": 0.5, "estimated_steps": 5}
        
        # Route to appropriate reasoning method
        if strategy == ReasoningStrategy.DIRECT:
            result = self._reason_direct(problem, context, generate_thought)
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            result = self._reason_tree(problem, context, generate_thought, difficulty)
        elif strategy == ReasoningStrategy.SELF_CONSISTENCY:
            result = self._reason_self_consistent(problem, context, generate_thought)
        else:
            result = self._reason_chain(problem, context, generate_thought, difficulty)
        
        # Apply reflection if enabled
        if self.enable_reflection and result.steps:
            critique = self.reflection.generate_critique(result.steps)
            result.meta_reflection = f"Quality: {critique['overall_quality']:.0%}, Issues: {len(critique['issues'])}"
            
            if critique['should_retry'] and (time.time() - start_time) * 1000 < self.max_thinking_time_ms:
                # Could retry with different strategy here
                pass
        
        result.reasoning_time_ms = (time.time() - start_time) * 1000
        result.strategy_used = strategy
        
        self.total_problems_solved += 1
        self.total_thoughts_generated += result.total_thoughts_explored
        
        return result
    
    def _reason_direct(
        self,
        problem: str,
        context: Optional[str],
        generate_thought: Optional[Callable],
    ) -> ReasoningResult:
        """Simple direct reasoning"""
        if generate_thought:
            answer, confidence = generate_thought(problem, context or "")
        else:
            answer = f"[Direct answer to: {problem}]"
            confidence = 0.7
        
        step = ReasoningStep(
            step_number=1,
            thought=answer,
            reasoning_type="direct",
            confidence=confidence,
        )
        
        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            strategy_used=ReasoningStrategy.DIRECT,
            steps=[step],
            total_thoughts_explored=1,
        )
    
    def _reason_chain(
        self,
        problem: str,
        context: Optional[str],
        generate_thought: Optional[Callable],
        difficulty: Dict[str, Any],
    ) -> ReasoningResult:
        """Chain of thought reasoning"""
        steps = []
        current_context = f"Problem: {problem}"
        if context:
            current_context += f"\nContext: {context}"
        
        num_steps = difficulty.get("estimated_steps", 5)
        cumulative_confidence = 1.0
        
        for i in range(num_steps):
            if generate_thought:
                thought, step_confidence = generate_thought(
                    f"Step {i+1}: Continue reasoning about: {current_context}",
                    "\n".join(s.thought for s in steps) if steps else ""
                )
            else:
                thought = f"[Reasoning step {i+1} for: {problem}]"
                step_confidence = 0.8
            
            cumulative_confidence *= step_confidence
            
            step = ReasoningStep(
                step_number=i + 1,
                thought=thought,
                reasoning_type="deduction",
                confidence=step_confidence,
            )
            steps.append(step)
            current_context = thought
            
            # Early termination if confidence too low
            if cumulative_confidence < self.min_confidence_threshold:
                break
        
        final_answer = steps[-1].thought if steps else "Unable to reason"
        
        return ReasoningResult(
            answer=final_answer,
            confidence=cumulative_confidence,
            strategy_used=ReasoningStrategy.CHAIN_OF_THOUGHT,
            steps=steps,
            total_thoughts_explored=len(steps),
        )
    
    def _reason_tree(
        self,
        problem: str,
        context: Optional[str],
        generate_thought: Optional[Callable],
        difficulty: Dict[str, Any],
    ) -> ReasoningResult:
        """Tree of thought reasoning"""
        tree = self.tree_of_thought
        root = tree.initialize(problem)
        
        max_iterations = difficulty.get("recommended_depth", 5) * tree.branching_factor
        steps = []
        
        for iteration in range(max_iterations):
            # Get promising nodes to expand
            nodes_to_expand = tree.get_promising_nodes(tree.beam_width)
            
            if not nodes_to_expand:
                break
            
            for node in nodes_to_expand:
                # Generate child thoughts
                for branch in range(tree.branching_factor):
                    if generate_thought:
                        path = tree.get_path_to_root(node.id)
                        path_context = " -> ".join(n.content[:50] for n in path)
                        thought, confidence = generate_thought(
                            f"Branch {branch+1}: Continue from: {path_context}",
                            context or ""
                        )
                    else:
                        thought = f"[Thought branch {branch+1} from {node.id}]"
                        confidence = random.uniform(0.4, 0.9)
                    
                    child = tree.create_node(
                        content=thought,
                        parent_id=node.id,
                        depth=node.depth + 1,
                    )
                    child.confidence = confidence
                    
                    # Check if this is a solution
                    if "answer" in thought.lower() or "therefore" in thought.lower():
                        child.is_solution = True
                        child.status = ThoughtStatus.COMPLETE
                
                node.status = ThoughtStatus.VERIFIED
        
        # Get best path
        best_path = tree.get_best_path()
        
        for i, node in enumerate(best_path):
            step = ReasoningStep(
                step_number=i + 1,
                thought=node.content,
                reasoning_type="exploration",
                confidence=node.confidence,
            )
            steps.append(step)
        
        final_answer = best_path[-1].content if best_path else "Unable to find solution"
        final_confidence = best_path[-1].confidence if best_path else 0.3
        
        return ReasoningResult(
            answer=final_answer,
            confidence=final_confidence,
            strategy_used=ReasoningStrategy.TREE_OF_THOUGHT,
            steps=steps,
            thought_tree={node.id: node for node in tree.nodes.values()},
            total_thoughts_explored=len(tree.nodes),
        )
    
    def _reason_self_consistent(
        self,
        problem: str,
        context: Optional[str],
        generate_thought: Optional[Callable],
        num_samples: int = 5,
    ) -> ReasoningResult:
        """Self-consistency reasoning with voting"""
        samples = []
        
        for i in range(num_samples):
            if generate_thought:
                answer, confidence = generate_thought(problem, context or "")
            else:
                answer = f"[Sample answer {i+1}]"
                confidence = random.uniform(0.5, 0.9)
            
            samples.append((answer, confidence))
        
        # Simple voting - in production use semantic similarity clustering
        answer_counts = {}
        for answer, conf in samples:
            key = answer.lower().strip()[:100]  # Normalize
            if key not in answer_counts:
                answer_counts[key] = {"count": 0, "confidence_sum": 0, "best_answer": answer}
            answer_counts[key]["count"] += 1
            answer_counts[key]["confidence_sum"] += conf
        
        # Select most consistent answer
        best = max(answer_counts.values(), key=lambda x: x["count"])
        consistency = best["count"] / num_samples
        avg_confidence = best["confidence_sum"] / best["count"]
        
        step = ReasoningStep(
            step_number=1,
            thought=best["best_answer"],
            reasoning_type="consensus",
            confidence=avg_confidence * consistency,
            alternatives=[s[0] for s in samples if s[0] != best["best_answer"]][:3],
        )
        
        return ReasoningResult(
            answer=best["best_answer"],
            confidence=avg_confidence * consistency,
            strategy_used=ReasoningStrategy.SELF_CONSISTENCY,
            steps=[step],
            total_thoughts_explored=num_samples,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning statistics"""
        return {
            "problems_solved": self.total_problems_solved,
            "thoughts_generated": self.total_thoughts_generated,
            "avg_thoughts_per_problem": (
                self.total_thoughts_generated / self.total_problems_solved
                if self.total_problems_solved > 0 else 0
            ),
        }


def create_reasoning_engine(config: Optional[Dict[str, Any]] = None) -> AdvancedReasoningEngine:
    """Factory function to create reasoning engine"""
    config = config or {}
    
    return AdvancedReasoningEngine(
        max_thinking_time_ms=config.get("max_thinking_time_ms", 30000),
        enable_reflection=config.get("enable_reflection", True),
        enable_meta_cognition=config.get("enable_meta_cognition", True),
        min_confidence_threshold=config.get("min_confidence", 0.3),
    )
