use std::sync::{Arc, Mutex};

use crate::numeric_id::NumericId;

use crate::{
    common::Value,
    table_spec::{ColumnId, Constraint, Table},
};

use super::{DisplacedTable, LeaderChange};

fn v(x: usize) -> Value {
    Value::from_usize(x)
}

#[test]
fn displaced() {
    empty_execution_state!(e);
    let mut d = DisplacedTable::default();
    {
        let mut buf = d.new_buffer();
        buf.stage_insert(&[v(0), v(1), v(0)]);
        buf.stage_insert(&[v(2), v(3), v(0)]);
    }
    d.merge(&mut e);
    let all = d.all();
    let mut updates = Vec::new();
    d.scan_generic(all.as_ref(), |_, row| {
        assert_eq!(row[2], v(0));
        updates.push((row[0], row[1]))
    });
    assert_eq!(updates.len(), 2);
    assert_ne!(updates[0], updates[1]);
    let eq_fst = d.refine(
        all,
        &[Constraint::EqConst {
            col: ColumnId::new(0),
            val: updates[0].0,
        }],
    );
    let mut rows = Vec::new();
    d.scan_generic(eq_fst.as_ref(), |_, row| {
        assert_eq!(row.len(), 3);
        rows.push((row[0], row[1], row[2]))
    });
    assert_eq!(rows, vec![(updates[0].0, updates[0].1, v(0))]);

    d.new_buffer().stage_insert(&[v(1), v(3), v(1)]);
    d.merge(&mut e);

    let all = d.all();
    let mut updates_2 = Vec::new();
    d.scan_generic(all.as_ref(), |_, row| updates_2.push((row[0], row[1])));
    assert!(updates_2.windows(2).all(|x| x[0].1 == x[1].1));
}

#[test]
fn displaced_leader_change_callback() {
    empty_execution_state!(e);
    let changes: Arc<Mutex<Vec<LeaderChange>>> = Arc::new(Mutex::new(Vec::new()));
    let changes_ref = Arc::clone(&changes);
    let mut d = DisplacedTable::with_leader_change_callback(move |_, change| {
        changes_ref.lock().unwrap().push(change);
    });
    {
        let mut buf = d.new_buffer();
        buf.stage_insert(&[v(5), v(3), v(0)]);
        buf.stage_insert(&[v(5), v(3), v(1)]);
    }
    d.merge(&mut e);

    {
        let changes = changes.lock().unwrap();
        assert_eq!(changes.len(), 1);
        let change = changes[0];
        assert_eq!(change.write_lhs, v(5));
        assert_eq!(change.lhs_leader, v(5));
        assert_eq!(change.write_rhs, v(3));
        assert_eq!(change.rhs_leader, v(3));
        assert_eq!(change.ts, v(0));
        assert_eq!(change.new_leader(), v(3));
    }

    {
        let mut buf = d.new_buffer();
        buf.stage_insert(&[v(5), v(2), v(2)]);
    }
    d.merge(&mut e);

    let changes = changes.lock().unwrap();
    assert_eq!(changes.len(), 2);
    let change = changes[1];
    assert_eq!(change.write_lhs, v(5));
    assert_eq!(change.lhs_leader, v(3));
    assert_eq!(change.write_rhs, v(2));
    assert_eq!(change.rhs_leader, v(2));
    assert_eq!(change.ts, v(2));
    assert_eq!(change.new_leader(), v(2));
}
