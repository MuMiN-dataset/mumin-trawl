CREATE CONSTRAINT reviewerConstraint IF NOT EXISTS ON (n:Reviewer) ASSERT n.url IS UNIQUE
