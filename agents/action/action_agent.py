"""
Action Agent

Executes decisions: database write-backs, letter generation, API calls, transactions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2

from common.schemas import ActionRequest, ActionResponse

logger = logging.getLogger(__name__)


class ActionAgent:
    """
    Action Agent for executing decisions

    Responsibilities:
    - Write decision outcomes to source systems (ERP, CRM, etc.)
    - Generate letters and notifications
    - Call external APIs for actions
    - Execute financial transactions
    - Maintain audit trail of all actions
    """

    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize Action Agent

        Args:
            db_connection_string: PostgreSQL connection string for audit trail
            output_dir: Directory for generated files (letters, reports)
        """
        self.db_connection_string = db_connection_string
        self.output_dir = output_dir or Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database connection if provided
        self.db_conn = None
        if db_connection_string:
            try:
                self.db_conn = psycopg2.connect(db_connection_string)
                logger.info("✅ Action Agent connected to database")
            except Exception as e:
                logger.warning(f"Failed to connect to database: {e}")

        logger.info(f"✅ Action Agent initialized with output dir: {output_dir}")

    def execute(
        self,
        request: ActionRequest,
    ) -> ActionResponse:
        """
        Execute action based on policy decision

        Args:
            request: ActionRequest with action type and parameters

        Returns:
            ActionResponse with execution results
        """
        case_id = request.case_id
        action_type = request.action_type
        action_params = request.action_params

        try:
            # Route to appropriate action handler
            if action_type == "write_back":
                result = self._write_back(case_id, action_params)

            elif action_type == "generate_letter":
                result = self._generate_letter(case_id, action_params)

            elif action_type == "send_notification":
                result = self._send_notification(case_id, action_params)

            elif action_type == "execute_transaction":
                result = self._execute_transaction(case_id, action_params)

            elif action_type == "api_call":
                result = self._api_call(case_id, action_params)

            elif action_type == "escalate":
                result = self._escalate(case_id, action_params)

            else:
                logger.warning(f"Unknown action type: {action_type}")
                result = {
                    "status": "failed",
                    "error": f"Unknown action type: {action_type}",
                }

            # Log action to audit trail
            self._log_audit_trail(
                case_id=case_id,
                action_type=action_type,
                action_params=action_params,
                result=result,
            )

            success = result.get("status") == "success"
            explanation = result.get("message", f"Action {action_type} executed")

            logger.info(
                f"Action executed for case {case_id}: {action_type} "
                f"(success={success})"
            )

            return ActionResponse(
                case_id=case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                success=success,
                action_type=action_type,
                result=result,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"Action execution failed for case {case_id}: {e}")

            # Log failure to audit trail
            self._log_audit_trail(
                case_id=case_id,
                action_type=action_type,
                action_params=action_params,
                result={"status": "failed", "error": str(e)},
            )

            return ActionResponse(
                case_id=case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                success=False,
                action_type=action_type,
                result={"error": str(e)},
                explanation=f"Action error: {str(e)}",
            )

    def _write_back(
        self,
        case_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Write decision back to source system

        Args:
            case_id: Case ID
            params: Write-back parameters

        Returns:
            Result dict
        """
        target_system = params.get("target_system", "database")
        entity_type = params.get("entity_type")
        entity_id = params.get("entity_id")
        updates = params.get("updates", {})

        logger.info(
            f"Writing back to {target_system}: {entity_type}/{entity_id}"
        )

        if target_system == "database":
            # Write to database
            if not self.db_conn:
                return {
                    "status": "failed",
                    "error": "Database connection not configured",
                }

            try:
                cursor = self.db_conn.cursor()

                # Build UPDATE query
                set_clause = ", ".join(
                    f"{key} = %s" for key in updates.keys()
                )
                query = f"""
                    UPDATE {entity_type}
                    SET {set_clause}, updated_at = NOW()
                    WHERE id = %s
                """

                values = list(updates.values()) + [entity_id]
                cursor.execute(query, values)
                self.db_conn.commit()

                return {
                    "status": "success",
                    "message": f"Updated {entity_type}/{entity_id}",
                    "rows_affected": cursor.rowcount,
                }

            except Exception as e:
                self.db_conn.rollback()
                logger.error(f"Database write-back failed: {e}")
                return {
                    "status": "failed",
                    "error": str(e),
                }

        elif target_system == "api":
            # Call external API
            return self._api_call(case_id, params)

        else:
            return {
                "status": "failed",
                "error": f"Unsupported target system: {target_system}",
            }

    def _generate_letter(
        self,
        case_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate letter or document

        Args:
            case_id: Case ID
            params: Letter parameters

        Returns:
            Result dict
        """
        letter_type = params.get("letter_type", "generic")
        recipient = params.get("recipient", {})
        content_vars = params.get("content_vars", {})

        logger.info(f"Generating {letter_type} letter for case {case_id}")

        # Load template
        template_path = Path(__file__).parent / "templates" / f"{letter_type}.txt"

        if template_path.exists():
            template = template_path.read_text()
        else:
            # Fallback template
            template = self._get_default_template(letter_type)

        # Substitute variables
        content = template
        for key, value in content_vars.items():
            content = content.replace(f"{{{{{key}}}}}", str(value))

        # Generate output file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"letter_{case_id}_{timestamp}.txt"

        output_file.write_text(content)

        return {
            "status": "success",
            "message": f"Letter generated: {output_file}",
            "file_path": str(output_file),
            "recipient": recipient,
        }

    def _send_notification(
        self,
        case_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Send notification (email, SMS, webhook)

        Args:
            case_id: Case ID
            params: Notification parameters

        Returns:
            Result dict
        """
        notification_type = params.get("notification_type", "email")
        recipient = params.get("recipient")
        subject = params.get("subject", f"Case {case_id} Update")
        message = params.get("message", "")

        logger.info(
            f"Sending {notification_type} notification for case {case_id}"
        )

        # In production, integrate with email/SMS service
        # For now, log to file
        notification_file = self.output_dir / f"notification_{case_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        notification_data = {
            "case_id": case_id,
            "type": notification_type,
            "recipient": recipient,
            "subject": subject,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }

        notification_file.write_text(json.dumps(notification_data, indent=2))

        return {
            "status": "success",
            "message": f"Notification sent (simulated): {notification_file}",
            "notification_file": str(notification_file),
        }

    def _execute_transaction(
        self,
        case_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute financial transaction

        Args:
            case_id: Case ID
            params: Transaction parameters

        Returns:
            Result dict
        """
        transaction_type = params.get("transaction_type")
        amount = params.get("amount", 0.0)
        from_account = params.get("from_account")
        to_account = params.get("to_account")
        reference = params.get("reference", case_id)

        logger.info(
            f"Executing {transaction_type} transaction for case {case_id}: "
            f"${amount} from {from_account} to {to_account}"
        )

        # In production, integrate with payment gateway
        # For now, log transaction
        transaction_file = self.output_dir / f"transaction_{case_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        transaction_data = {
            "case_id": case_id,
            "type": transaction_type,
            "amount": amount,
            "from_account": from_account,
            "to_account": to_account,
            "reference": reference,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending",
        }

        transaction_file.write_text(json.dumps(transaction_data, indent=2))

        return {
            "status": "success",
            "message": f"Transaction logged (simulated): {transaction_file}",
            "transaction_id": f"TXN_{case_id}_{datetime.utcnow().timestamp()}",
            "transaction_file": str(transaction_file),
        }

    def _api_call(
        self,
        case_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call external API

        Args:
            case_id: Case ID
            params: API call parameters

        Returns:
            Result dict
        """
        endpoint = params.get("endpoint")
        method = params.get("method", "POST")
        payload = params.get("payload", {})
        headers = params.get("headers", {})

        logger.info(f"Calling API {method} {endpoint} for case {case_id}")

        # In production, use requests library
        # For now, log API call
        api_call_file = self.output_dir / f"api_call_{case_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        api_call_data = {
            "case_id": case_id,
            "endpoint": endpoint,
            "method": method,
            "payload": payload,
            "headers": headers,
            "timestamp": datetime.utcnow().isoformat(),
        }

        api_call_file.write_text(json.dumps(api_call_data, indent=2))

        return {
            "status": "success",
            "message": f"API call logged (simulated): {api_call_file}",
            "api_call_file": str(api_call_file),
        }

    def _escalate(
        self,
        case_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Escalate case to human reviewer

        Args:
            case_id: Case ID
            params: Escalation parameters

        Returns:
            Result dict
        """
        escalation_reason = params.get("reason", "Policy decision requires escalation")
        assigned_to = params.get("assigned_to", "supervisor")
        priority = params.get("priority", "medium")

        logger.info(f"Escalating case {case_id} to {assigned_to}")

        escalation_file = self.output_dir / f"escalation_{case_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        escalation_data = {
            "case_id": case_id,
            "reason": escalation_reason,
            "assigned_to": assigned_to,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat(),
        }

        escalation_file.write_text(json.dumps(escalation_data, indent=2))

        return {
            "status": "success",
            "message": f"Case escalated to {assigned_to}",
            "escalation_file": str(escalation_file),
        }

    def _log_audit_trail(
        self,
        case_id: str,
        action_type: str,
        action_params: Dict[str, Any],
        result: Dict[str, Any],
    ):
        """
        Log action to audit trail

        Args:
            case_id: Case ID
            action_type: Action type
            action_params: Action parameters
            result: Execution result
        """
        audit_file = self.output_dir / "audit_trail.jsonl"

        audit_entry = {
            "case_id": case_id,
            "action_type": action_type,
            "action_params": action_params,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

        logger.debug(f"Audit trail logged for case {case_id}")

    def _get_default_template(self, letter_type: str) -> str:
        """
        Get default letter template

        Args:
            letter_type: Letter type

        Returns:
            Template string
        """
        if letter_type == "approval":
            return """
Dear {{recipient_name}},

We are pleased to inform you that your request for case {{case_id}} has been approved.

Transaction Details:
- Amount: {{transaction_amount}}
- Reference: {{reference}}
- Date: {{decision_date}}

If you have any questions, please contact us.

Sincerely,
QURE System
"""

        elif letter_type == "rejection":
            return """
Dear {{recipient_name}},

After careful review, we regret to inform you that your request for case {{case_id}} has been denied.

Reason: {{rejection_reason}}

If you believe this decision was made in error, you may appeal within 30 days.

Sincerely,
QURE System
"""

        else:
            return """
Dear {{recipient_name}},

This letter is regarding case {{case_id}}.

{{message}}

Sincerely,
QURE System
"""

    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed")


# Singleton instance
_action_agent: Optional[ActionAgent] = None


def get_action_agent(
    db_connection_string: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> ActionAgent:
    """
    Get or create singleton ActionAgent instance

    Args:
        db_connection_string: Database connection string
        output_dir: Output directory

    Returns:
        ActionAgent instance
    """
    global _action_agent

    if _action_agent is None:
        _action_agent = ActionAgent(
            db_connection_string=db_connection_string,
            output_dir=output_dir,
        )

    return _action_agent
