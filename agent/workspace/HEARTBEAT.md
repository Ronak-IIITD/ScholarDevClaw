# ScholarDevClaw Heartbeat

## On Each Wake

1. Check for pending integration tasks in Convex
2. Resume any paused integration workflows
3. Process any approval responses from users
4. Run validation benchmarks for long-running tasks
5. Update integration status and logs

## Task Processing

### Check for Pending Work
```
Query Convex for integrations with status 'pending'
For each:
  - Validate input (repo URL, paper source)
  - Begin Phase 1
```

### Resume Paused Workflows
```
Find integrations where status starts with 'phase*'
Continue from last completed phase
If step_approval mode, wait for approval before proceeding
```

### Handle Approvals
```
Check for integrations awaiting approval
If approved:
  - Continue to next phase
If rejected:
  - Mark as failed with reason
  - Generate failure report
```

## Long-running Operations

For validation benchmarks:
- Spawn background process
- Update progress in Convex
- On next heartbeat, check if complete
- If complete, process results and continue

## Error Handling

On any failure:
1. Log error with full context
2. Save current state to Convex
3. If retry count < 2, schedule retry
4. Otherwise, mark as failed and notify user

## Exit Conditions

- No pending work
- All integrations complete or failed
- No approvals waiting
