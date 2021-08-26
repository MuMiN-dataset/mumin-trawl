UNWIND $polls as poll
WITH poll
WHERE poll.id IS NOT NULL
MERGE (n:Poll {pollId:poll.id})
ON CREATE SET
    n.durationMinutes = toInteger(poll.duration_minutes),
    n.endDatetime = poll.end_datetime,
    n.votingStatus = poll.voting_status,
    n.options = poll.options;
