import torch


class Query(torch.Tensor):
    """Tensor storage of logical queries in postfix notations."""

    projection = 1 << 58
    intersection = 1 << 59
    union = 1 << 60
    negation = 1 << 61
    stop = 1 << 62
    operation = projection | intersection | union | negation | stop

    stack_size = 2

    def __new__(cls, data, device=None):
        query = torch.as_tensor(data, dtype=torch.long, device=device)
        query = torch.Tensor._make_subclass(cls, query)
        return query

    @classmethod
    def from_nested(cls, nested, binary_op=True):
        """Construct a logical query from nested tuples (BetaE format)."""
        if not binary_op:
            raise ValueError("The current implementation doesn't support nary operations")
        query = cls.nested_to_postfix(nested, binary_op=binary_op)
        query.append(cls.stop)
        return cls(query)

    @classmethod
    def nested_to_postfix(cls, nested, binary_op=True):
        """Recursively convert nested tuples into a postfix notation."""
        query = []

        if len(nested) == 2 and isinstance(nested[-1][-1], int):
            var, unary_ops = nested
            if isinstance(var, tuple):
                query += cls.nested_to_postfix(var, binary_op=binary_op)
            else:
                query.append(var)
            for op in unary_ops:
                if op == -2:
                    query.append(cls.negation)
                else:
                    query.append(cls.projection | op)
        else:
            if len(nested[-1]) > 1:
                vars, nary_op = nested, cls.intersection
            else:
                vars, nary_op = nested[:-1], cls.union
            num_args = 2 if binary_op else len(vars)
            op = nary_op | num_args
            for i, var in enumerate(vars):
                query += cls.nested_to_postfix(var)
                if i + 1 >= num_args:
                    query.append(op)

        return query

    def to_readable(self):
        """Convert this logical query to a human readable string."""
        if self.ndim > 1:
            raise ValueError("readable() can only be called for a single query")

        num_variable = 0
        stack = []
        lines = []
        for op in self:
            if op.is_operand():
                entity = op.get_operand().item()
                stack.append(str(entity))
            else:
                var = chr(ord("A") + num_variable)
                if op.is_projection():
                    relation = op.get_operand().item()
                    line = "%s <- projection_%d(%s)" % (var, relation, stack.pop())
                elif op.is_intersection():
                    num_args = op.get_operand()
                    args = stack[-num_args:]
                    stack = stack[:-num_args]
                    line = "%s <- intersection(%s)" % (var, ", ".join(args))
                elif op.is_union():
                    num_args = op.get_operand().item()
                    args = stack[-num_args:]
                    stack = stack[:-num_args]
                    line = "%s <- union(%s, %s)" % (var, ", ".join(args))
                elif op.is_negation():
                    line = "%s <- negation(%s)" % (var, stack.pop())
                elif op.is_stop():
                    break
                else:
                    raise ValueError("Unknown operator `%d`" % op)
                lines.append(line)
                stack.append(var)
                num_variable += 1

        if len(stack) > 1:
            raise ValueError("Invalid query. More operands than expected")
        line = "\n".join(lines)
        return line

    def computation_graph(self):
        """Get the computation graph of logical queries. Used for visualization."""
        query = self.view(-1, self.shape[-1])
        stack = Stack(len(query), self.stack_size, dtype=torch.long, device=query.device)
        # pointer to the next operator that consumes the output of this operator
        pointer = -torch.ones(query.shape, dtype=torch.long, device=query.device)
        # depth of each operator in the computation graph
        depth = -torch.ones(query.shape, dtype=torch.long, device=query.device)
        # width of the substree covered by each operator
        width = -torch.ones(query.shape, dtype=torch.long, device=query.device)

        for i, op in enumerate(query.t()):
            is_operand = op.is_operand()
            is_unary = op.is_projection() | op.is_negation()
            is_binary = op.is_intersection() | op.is_union()
            is_stop = op.is_stop()
            if is_operand.any():
                stack.push(is_operand, i)
                depth[is_operand, i] = 0
                width[is_operand, i] = 1
            if is_unary.any():
                prev = stack.pop(is_unary)
                pointer[is_unary, prev] = i
                depth[is_unary, i] = depth[is_unary, prev] + 1
                width[is_unary, i] = width[is_unary, prev]
                stack.push(is_unary, i)
            if is_binary.any():
                prev_y = stack.pop(is_binary)
                prev_x = stack.pop(is_binary)
                pointer[is_binary, prev_y] = i
                pointer[is_binary, prev_x] = i
                depth[is_binary, i] = torch.max(depth[is_binary, prev_x], depth[is_binary, prev_y]) + 1
                width[is_binary, i] = width[is_binary, prev_x] + width[is_binary, prev_y]
                stack.push(is_binary, i)
            if is_stop.all():
                break

        # each operator covers leaf nodes [left, right)
        left = torch.where(depth > 0, 0, -1)
        right = torch.where(depth > 0, width.max(), -1)
        # backtrack to update left and right
        for i in reversed(range(query.shape[-1])):
            has_pointer = pointer[:, i] != -1
            ptr = pointer[has_pointer, i]
            depth[has_pointer, i] = depth[has_pointer, ptr] - 1
            left[has_pointer, i] = left[has_pointer, ptr] + width[has_pointer, ptr] - width[has_pointer, i]
            right[has_pointer, i] = left[has_pointer, i] + width[has_pointer, i]
            width[has_pointer, ptr] -= width[has_pointer, i]

        pointer = pointer.view_as(self)
        depth = depth.view_as(self)
        left = left.view_as(self)
        right = right.view_as(self)
        return pointer, depth, left, right

    def is_operation(self):
        return (self & self.operation > 0)

    def is_operand(self):
        return ~(self & self.operation > 0)

    def is_projection(self):
        return self & self.projection > 0

    def is_intersection(self):
        return self & self.intersection > 0

    def is_union(self):
        return self & self.union > 0

    def is_negation(self):
        return self & self.negation > 0

    def is_stop(self):
        return self & self.stop > 0

    def get_operation(self):
        return self & self.operation

    def get_operand(self):
        return self & ~self.operation

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class Stack(object):
    """
    Batch of stacks implemented in PyTorch.

    Parameters:
        batch_size (int): batch size
        stack_size (int): max stack size
        shape (tuple of int, optional): shape of each element in the stack
        dtype (torch.dtype): dtype
        device (torch.device): device
    """

    def __init__(self, batch_size, stack_size, *shape, dtype=None, device=None):
        self.stack = torch.zeros(batch_size, stack_size, *shape, dtype=dtype, device=device)
        self.SP = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.batch_size = batch_size
        self.stack_size = stack_size

    def push(self, mask, value):
        if (self.SP[mask] >= self.stack_size).any():
            raise ValueError("Stack overflow")
        self.stack[mask, self.SP[mask]] = value
        self.SP[mask] += 1

    def pop(self, mask=None):
        if (self.SP[mask] < 1).any():
            raise ValueError("Stack underflow")
        if mask is None:
            mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.stack.device)
        self.SP[mask] -= 1
        return self.stack[mask, self.SP[mask]]

    def top(self, mask=None):
        if (self.SP < 1).any():
            raise ValueError("Stack is empty")
        if mask is None:
            mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.stack.device)
        return self.stack[mask, self.SP[mask] - 1]
