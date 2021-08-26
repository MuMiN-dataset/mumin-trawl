CREATE CONSTRAINT conversationConstraint IF NOT EXISTS
ON (n:Conversation) ASSERT n.conversationId IS UNIQUE
