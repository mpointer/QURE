"""
Orchestration Agent

Coordinates multi-agent workflow execution using DAG-based pipelines.
Now integrates with Planner QRU for intelligent, dynamic QRU selection.
"""

import logging
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from common.schemas import (
    AlgorithmRequest,
    AssuranceRequest,
    DataProcessingRequest,
    GenAIReasoningRequest,
    MLPredictionRequest,
    PolicyDecisionRequest,
    RulesEvaluationRequest,
    ActionRequest,
)
from agents.planner import PlannerQRU, ExecutionPlan

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class StepStatus(str, Enum):
    """Workflow step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Orchestrator:
    """
    Orchestration Agent for workflow coordination

    Responsibilities:
    - Define and execute DAG-based workflows
    - Route requests between agents
    - Handle dependencies and parallelism
    - Retry failed steps with exponential backoff
    - Monitor workflow progress
    - Support branching logic (if/else, switch)
    - Integrate with Planner QRU for intelligent QRU selection
    """

    def __init__(self):
        """Initialize Orchestrator"""
        # Workflow registry
        self.workflows: Dict[str, Dict[str, Any]] = {}

        # Active workflow instances
        self.instances: Dict[str, Dict[str, Any]] = {}

        # Agent registry (will be populated)
        self.agents: Dict[str, Any] = {}

        # Planner QRU for intelligent orchestration
        self.planner = PlannerQRU()

        # Execution plans (cached by case_id)
        self.execution_plans: Dict[str, ExecutionPlan] = {}

        logger.info("✅ Orchestrator initialized with Planner QRU")

    def register_workflow(
        self,
        workflow_name: str,
        workflow_definition: Dict[str, Any],
    ):
        """
        Register workflow definition

        Args:
            workflow_name: Workflow name
            workflow_definition: Workflow DAG definition
        """
        self.workflows[workflow_name] = workflow_definition
        logger.info(f"Registered workflow: {workflow_name}")

    def register_agent(self, agent_name: str, agent_instance: Any):
        """
        Register agent instance

        Args:
            agent_name: Agent name
            agent_instance: Agent instance
        """
        self.agents[agent_name] = agent_instance
        logger.debug(f"Registered agent: {agent_name}")

    def start_workflow(
        self,
        workflow_name: str,
        case_id: str,
        initial_data: Dict[str, Any],
    ) -> str:
        """
        Start workflow execution

        Args:
            workflow_name: Workflow name
            case_id: Case ID
            initial_data: Initial workflow data

        Returns:
            Workflow instance ID
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_name}")

        # Create workflow instance
        instance_id = f"{workflow_name}_{case_id}_{int(datetime.utcnow().timestamp())}"

        workflow_def = self.workflows[workflow_name]

        instance = {
            "instance_id": instance_id,
            "workflow_name": workflow_name,
            "case_id": case_id,
            "status": WorkflowStatus.PENDING,
            "data": initial_data,
            "steps": {},
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }

        # Initialize step statuses
        for step_name in workflow_def["steps"].keys():
            instance["steps"][step_name] = {
                "status": StepStatus.PENDING,
                "result": None,
                "error": None,
                "started_at": None,
                "completed_at": None,
            }

        self.instances[instance_id] = instance

        logger.info(f"Started workflow {workflow_name} instance: {instance_id}")

        return instance_id

    def start_intelligent_workflow(
        self,
        case_id: str,
        case_data: Dict[str, Any],
        vertical: str,
    ) -> str:
        """
        Start workflow with Planner QRU intelligence

        This method:
        1. Invokes Planner QRU to analyze the case
        2. Generates an optimized ExecutionPlan
        3. Dynamically constructs workflow based on selected QRUs
        4. Executes the workflow

        Args:
            case_id: Case ID
            case_data: Case data for analysis
            vertical: Business vertical (Finance, Healthcare, Insurance, etc.)

        Returns:
            Workflow instance ID
        """
        logger.info(f"🧠 Invoking Planner QRU for case {case_id} in {vertical} vertical")

        # Step 1: Invoke Planner QRU
        execution_plan = self.planner.analyze_case(case_data, vertical)

        # Cache execution plan
        self.execution_plans[case_id] = execution_plan

        logger.info(f"📋 Planner selected {len(execution_plan.selected_qrus)} QRUs")
        logger.info(f"💰 Estimated cost: ${execution_plan.estimated_total_cost:.4f}")
        logger.info(f"⏱️  Estimated time: {execution_plan.estimated_total_time_seconds:.1f}s")

        # Step 2: Build dynamic workflow definition from execution plan
        workflow_def = self._build_workflow_from_plan(execution_plan)

        # Register the dynamic workflow
        workflow_name = f"dynamic_{vertical}_{case_id}"
        self.register_workflow(workflow_name, workflow_def)

        # Step 3: Start workflow execution
        instance_id = self.start_workflow(
            workflow_name=workflow_name,
            case_id=case_id,
            initial_data=case_data,
        )

        # Add execution plan to instance (convert dataclass to dict)
        self.instances[instance_id]["execution_plan"] = asdict(execution_plan)

        return instance_id

    def _build_workflow_from_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Build workflow definition from ExecutionPlan

        Args:
            plan: ExecutionPlan from Planner QRU

        Returns:
            Workflow definition dict
        """
        steps = {}
        previous_step = None

        for qru_selection in plan.selected_qrus:
            qru_name = qru_selection.qru_name
            step_name = qru_name.lower().replace(" ", "_")

            # Map QRU names to agent names
            agent_name_map = {
                "Retriever": "retriever",
                "Data": "data_agent",
                "Rules": "rules_engine",
                "Algorithm": "algorithm_agent",
                "ML Model": "ml_model_agent",
                "GenAI": "genai_reasoner",
                "Assurance": "assurance_agent",
                "Policy": "policy_agent",
                "Action": "action_agent",
                "Learning": "learning_agent",
                "Orchestration": "orchestrator",
            }

            agent_name = agent_name_map.get(qru_name, qru_name.lower())

            step_def = {
                "agent": agent_name,
                "action": "execute",
                "inputs": {},
                "depends_on": [previous_step] if previous_step else [],
                "on_failure": "continue" if not qru_selection.required else "stop",
            }

            steps[step_name] = step_def
            previous_step = step_name

        return {
            "name": f"Dynamic workflow from Planner",
            "description": plan.reasoning,
            "steps": steps,
        }

    def execute_workflow(self, instance_id: str) -> Dict[str, Any]:
        """
        Execute workflow instance

        Args:
            instance_id: Workflow instance ID

        Returns:
            Workflow result dict
        """
        if instance_id not in self.instances:
            raise ValueError(f"Workflow instance not found: {instance_id}")

        instance = self.instances[instance_id]
        workflow_name = instance["workflow_name"]
        workflow_def = self.workflows[workflow_name]

        logger.info(f"Executing workflow instance: {instance_id}")

        instance["status"] = WorkflowStatus.RUNNING

        try:
            # Execute steps in topological order
            execution_order = self._topological_sort(workflow_def)

            for step_name in execution_order:
                step_def = workflow_def["steps"][step_name]

                # Check dependencies
                if not self._check_dependencies(instance, step_def):
                    logger.debug(f"Skipping step {step_name} (unmet dependencies)")
                    instance["steps"][step_name]["status"] = StepStatus.SKIPPED
                    continue

                # Execute step
                logger.info(f"Executing step: {step_name}")
                self._execute_step(instance, step_name, step_def)

                # Check if step failed
                if instance["steps"][step_name]["status"] == StepStatus.FAILED:
                    # Handle failure
                    if step_def.get("on_failure") == "continue":
                        logger.warning(f"Step {step_name} failed, continuing")
                        continue
                    else:
                        logger.error(f"Step {step_name} failed, stopping workflow")
                        instance["status"] = WorkflowStatus.FAILED
                        return instance

            # All steps completed
            instance["status"] = WorkflowStatus.COMPLETED
            instance["completed_at"] = datetime.utcnow().isoformat()

            logger.info(f"Workflow instance completed: {instance_id}")

            return instance

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            instance["status"] = WorkflowStatus.FAILED
            instance["error"] = str(e)
            return instance

    def _execute_step(
        self,
        instance: Dict[str, Any],
        step_name: str,
        step_def: Dict[str, Any],
    ):
        """
        Execute single workflow step

        Args:
            instance: Workflow instance
            step_name: Step name
            step_def: Step definition
        """
        step_state = instance["steps"][step_name]
        step_state["status"] = StepStatus.RUNNING
        step_state["started_at"] = datetime.utcnow().isoformat()

        agent_name = step_def["agent"]
        action = step_def.get("action", "execute")
        inputs = step_def.get("inputs", {})

        # Resolve input references from previous steps
        resolved_inputs = self._resolve_inputs(instance, inputs)

        try:
            # Get agent
            if agent_name not in self.agents:
                raise ValueError(f"Agent not found: {agent_name}")

            agent = self.agents[agent_name]

            # Call agent
            result = self._call_agent(
                agent=agent,
                agent_name=agent_name,
                action=action,
                inputs=resolved_inputs,
                case_id=instance["case_id"],
            )

            # Store result
            step_state["result"] = result
            step_state["status"] = StepStatus.COMPLETED
            step_state["completed_at"] = datetime.utcnow().isoformat()

            # Update workflow data
            output_key = step_def.get("output_key", step_name)
            instance["data"][output_key] = result

            logger.debug(f"Step {step_name} completed successfully")

        except Exception as e:
            logger.error(f"Step {step_name} failed: {e}")
            step_state["status"] = StepStatus.FAILED
            step_state["error"] = str(e)
            step_state["completed_at"] = datetime.utcnow().isoformat()

    def _call_agent(
        self,
        agent: Any,
        agent_name: str,
        action: str,
        inputs: Dict[str, Any],
        case_id: str,
    ) -> Any:
        """
        Call agent with inputs

        Args:
            agent: Agent instance
            agent_name: Agent name
            action: Action to perform
            inputs: Input parameters
            case_id: Case ID

        Returns:
            Agent result
        """
        # Build request based on agent type
        if "retriever" in agent_name:
            # Retriever doesn't use Pydantic schemas yet
            return agent.retrieve(inputs)

        elif "data" in agent_name:
            request = DataProcessingRequest(
                case_id=case_id,
                from_agent="orchestrator",
                to_agent="data_agent",
                **inputs,
            )
            response = agent.process(request)
            return response.dict()

        elif "rules" in agent_name:
            request = RulesEvaluationRequest(
                case_id=case_id,
                from_agent="orchestrator",
                to_agent="rules_engine",
                **inputs,
            )
            response = agent.evaluate(request)
            return response.dict()

        elif "algorithm" in agent_name:
            request = AlgorithmRequest(
                case_id=case_id,
                from_agent="orchestrator",
                to_agent="algorithm_agent",
                **inputs,
            )
            response = agent.execute(request)
            return response.dict()

        elif "ml_model" in agent_name or "ml" in agent_name:
            request = MLPredictionRequest(
                case_id=case_id,
                from_agent="orchestrator",
                to_agent="ml_model_agent",
                **inputs,
            )
            response = agent.predict(request)
            return response.dict()

        elif "genai" in agent_name:
            request = GenAIReasoningRequest(
                case_id=case_id,
                from_agent="orchestrator",
                to_agent="genai_reasoner",
                **inputs,
            )
            response = agent.reason(request)
            return response.dict()

        elif "assurance" in agent_name:
            request = AssuranceRequest(
                case_id=case_id,
                from_agent="orchestrator",
                to_agent="assurance_agent",
                **inputs,
            )
            response = agent.evaluate(request)
            return response.dict()

        elif "policy" in agent_name:
            request = PolicyDecisionRequest(
                case_id=case_id,
                from_agent="orchestrator",
                to_agent="policy_agent",
                **inputs,
            )
            response = agent.decide(request)
            return response.dict()

        elif "action" in agent_name:
            request = ActionRequest(
                case_id=case_id,
                from_agent="orchestrator",
                to_agent="action_agent",
                **inputs,
            )
            response = agent.execute(request)
            return response.dict()

        else:
            # Generic call
            if hasattr(agent, action):
                method = getattr(agent, action)
                return method(**inputs)
            else:
                raise ValueError(f"Agent {agent_name} does not have action: {action}")

    def _resolve_inputs(
        self,
        instance: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve input references from previous steps

        Args:
            instance: Workflow instance
            inputs: Input dict with potential references

        Returns:
            Resolved inputs dict
        """
        resolved = {}

        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to workflow data
                ref_key = value[1:]  # Remove $
                if ref_key in instance["data"]:
                    resolved[key] = instance["data"][ref_key]
                else:
                    logger.warning(f"Reference not found: {ref_key}")
                    resolved[key] = None
            else:
                resolved[key] = value

        return resolved

    def _check_dependencies(
        self,
        instance: Dict[str, Any],
        step_def: Dict[str, Any],
    ) -> bool:
        """
        Check if step dependencies are met

        Args:
            instance: Workflow instance
            step_def: Step definition

        Returns:
            True if dependencies met
        """
        depends_on = step_def.get("depends_on", [])

        for dep_step in depends_on:
            if dep_step not in instance["steps"]:
                return False

            dep_status = instance["steps"][dep_step]["status"]

            if dep_status != StepStatus.COMPLETED:
                return False

        return True

    def _topological_sort(
        self,
        workflow_def: Dict[str, Any],
    ) -> List[str]:
        """
        Topological sort of workflow steps

        Args:
            workflow_def: Workflow definition

        Returns:
            List of step names in execution order
        """
        steps = workflow_def["steps"]
        visited = set()
        order = []

        def visit(step_name: str):
            if step_name in visited:
                return

            step_def = steps[step_name]
            depends_on = step_def.get("depends_on", [])

            for dep in depends_on:
                visit(dep)

            visited.add(step_name)
            order.append(step_name)

        for step_name in steps.keys():
            visit(step_name)

        return order

    def get_workflow_status(self, instance_id: str) -> Dict[str, Any]:
        """
        Get workflow status

        Args:
            instance_id: Workflow instance ID

        Returns:
            Status dict
        """
        if instance_id not in self.instances:
            raise ValueError(f"Workflow instance not found: {instance_id}")

        return self.instances[instance_id]

    def pause_workflow(self, instance_id: str):
        """
        Pause workflow execution

        Args:
            instance_id: Workflow instance ID
        """
        if instance_id not in self.instances:
            raise ValueError(f"Workflow instance not found: {instance_id}")

        self.instances[instance_id]["status"] = WorkflowStatus.PAUSED
        logger.info(f"Paused workflow: {instance_id}")

    def resume_workflow(self, instance_id: str) -> Dict[str, Any]:
        """
        Resume paused workflow

        Args:
            instance_id: Workflow instance ID

        Returns:
            Workflow result dict
        """
        if instance_id not in self.instances:
            raise ValueError(f"Workflow instance not found: {instance_id}")

        instance = self.instances[instance_id]

        if instance["status"] != WorkflowStatus.PAUSED:
            raise ValueError(f"Workflow is not paused: {instance_id}")

        logger.info(f"Resuming workflow: {instance_id}")

        return self.execute_workflow(instance_id)

    def get_execution_plan(self, case_id: str) -> Optional[ExecutionPlan]:
        """
        Get cached execution plan for a case

        Args:
            case_id: Case ID

        Returns:
            ExecutionPlan or None if not found
        """
        return self.execution_plans.get(case_id)


# Singleton instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """
    Get or create singleton Orchestrator instance

    Returns:
        Orchestrator instance
    """
    global _orchestrator

    if _orchestrator is None:
        _orchestrator = Orchestrator()

    return _orchestrator
